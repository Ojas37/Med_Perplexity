import json
import re
import os
import difflib
from typing import List, Dict, Any, Optional
from Bio import Entrez
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL", "pratham16salgaonkar@gmail.com")
Entrez.tool = "MedPerplexity_ResearchAgent"

JAN_AUSHADHI_DB_PATH = "data/jan_aushadhi.json"

# ==========================================
# 🛠 TOOL 1: PUBMED RESEARCHER
# ==========================================

def clean_xml_text(text: str) -> str:
    """Removes HTML Tags (<b>, <i>, <sup>) and extra whitespace."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    return " ".join(clean.split())

def get_pub_date(article_data: Dict) -> str:
    """Extracts publication date with a fallback to Electronic Date."""
    try:
        # 1. Try Standard Print Date
        pub_date_tag = article_data.get("MedlineCitation", {}).get("Article", {}).get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        if "Year" in pub_date_tag:
            return f"{pub_date_tag['Year']} {pub_date_tag.get('Month', '')} {pub_date_tag.get('Day', '')}".strip()
        if "MedlineDate" in pub_date_tag:
            return pub_date_tag["MedlineDate"]

        # 2. Try Electronic Date (Backup)
        article_date_list = article_data.get("MedlineCitation", {}).get("Article", {}).get("ArticleDate", [])
        if article_date_list and isinstance(article_date_list, list):
            date_data = article_date_list[0]
            if "Year" in date_data:
                return f"{date_data['Year']} {date_data.get('Month', '')} {date_data.get('Day', '')} (Epub)".strip()
        return "Unknown Date"
    except Exception:
        return "Unknown Date"

def extract_abstract_text(article_data: Dict) -> str:
    """Parses complex PubMed XML abstracts and cleans them."""
    try:
        abstract_obj = article_data.get("MedlineCitation", {}).get("Article", {}).get("Abstract", {})
        abstract_text_list = abstract_obj.get("AbstractText", [])
        
        if not abstract_text_list:
            return "No Abstract Available."

        combined_text = ""
        
        # Handle List of Sections (Structured Abstract)
        if isinstance(abstract_text_list, list):
            full_text = []
            for section in abstract_text_list:
                text_content = str(section)
                if hasattr(section, "attributes") and "Label" in section.attributes:
                    label = section.attributes["Label"]
                    text_content = f"{label.upper()}: {text_content}"
                full_text.append(text_content)
            combined_text = "\n".join(full_text)
        elif isinstance(abstract_text_list, str):
            combined_text = abstract_text_list
        else:
            combined_text = str(abstract_text_list)

        return clean_xml_text(combined_text)
    except Exception as e:
        return f"Error parsing abstract: {str(e)}"

def search_pubmed_ids(query: str, strict_mode: bool = True) -> List[str]:
    """Finds PMIDs for a query."""
    # 1. Date Filter (Last ~10 years to Future)
    date_filter = ' AND ("2015/01/01"[Date - Publication] : "3000"[Date - Publication])'
    # 2. Quality Filter
    quality_filter = ' AND (Systematic Review[pt] OR Guideline[pt] OR Clinical Trial[pt] OR Meta-Analysis[pt])' if strict_mode else ""
    # 3. Language Filter
    language_filter = " AND English[Language]"
    
    final_query = f"({query}){quality_filter}{date_filter}{language_filter}"
    
    try:
        handle = Entrez.esearch(db="pubmed", term=final_query, retmax=5, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        print(f"Error in Search: {e}")
        return []

def fetch_article_details(id_list: List[str]) -> List[Dict[str, Any]]:
    """Fetches full details for a list of PMIDs."""
    if not id_list: return []
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
        xml_data = Entrez.read(handle)
        handle.close()
        
        formatted_articles = []
        if "PubmedArticle" in xml_data:
            for article in xml_data["PubmedArticle"]:
                medline = article["MedlineCitation"]
                article_info = medline["Article"]
                
                title = clean_xml_text(article_info.get("ArticleTitle", "No Title"))
                pub_date = get_pub_date(article)
                journal = article_info.get("Journal", {}).get("ISOAbbreviation", "") or article_info.get("Journal", {}).get("Title", "Unknown Journal")
                abstract = extract_abstract_text(article)
                
                formatted_articles.append({
                    "pmid": str(medline["PMID"]),
                    "title": title,
                    "journal": journal,
                    "pub_date": pub_date,
                    "abstract": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{medline['PMID']}/"
                })
        return formatted_articles
    except Exception as e:
        print(f"Error in Fetch: {e}")
        return []

def query_pubmed_realtime(query: str) -> str:
    """
    Main entry point for Agents. Returns JSON string of results.
    """
    print(f"🔎 Researching: {query}...")
    pmids = search_pubmed_ids(query, strict_mode=True)
    
    if not pmids:
        print("⚠ No high-evidence results. Broadening search...")
        pmids = search_pubmed_ids(query, strict_mode=False)
    
    if not pmids:
        return json.dumps({"status": "error", "message": "No results found."})
        
    articles = fetch_article_details(pmids)
    return json.dumps({"status": "success", "evidence_count": len(articles), "articles": articles}, indent=2)


# ==========================================
# 🛠 TOOL 2: JAN AUSHADHI FINDER
# ==========================================

def get_similarity_score(query: str, target: str) -> float:
    """Calculates similarity score with substring bonus."""
    q = query.lower().strip()
    t = target.lower().strip()
    
    # Standard Fuzzy Score
    matcher = difflib.SequenceMatcher(None, q, t)
    base_score = matcher.ratio() * 100
    
    # Substring Bonus (e.g. "Atorvastatin" inside "Atorvastatin Calcium")
    if len(q) > 4 and q in t:
        return max(base_score, 95.0)
        
    return base_score

def load_jan_aushadhi_db(filepath: str = JAN_AUSHADHI_DB_PATH) -> List[Dict]:
    """Helper to load the JSON DB safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Database file {filepath} not found.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {filepath}.")
        return []

def search_jan_aushadhi(query_drug: str, medicine_database: Optional[List[Dict]] = None) -> Dict:
    """
    Searches for a drug in the database using fuzzy + substring matching.
    If database is not provided, it loads from default path.
    """
    if medicine_database is None:
        medicine_database = load_jan_aushadhi_db()
        
    if not medicine_database:
        return {"found": False, "message": "Database unavailable."}

    best_match = None
    highest_score = 0
    matched_name_source = "" 
    THRESHOLD = 85.0
    
    for record in medicine_database:
        # 1. Check Generic Name
        generic_name = record.get('generic_name', '')
        generic_score = get_similarity_score(query_drug, generic_name)
        
        # 2. Check Common Brand Names
        brand_score = 0
        current_best_brand = ""
        
        for brand in record.get('common_brands', []):
            b_score = get_similarity_score(query_drug, brand)
            if b_score > brand_score:
                brand_score = b_score
                current_best_brand = brand
        
        # 3. Determine best score for this record
        if generic_score > brand_score:
            current_record_score = generic_score
            current_source = f"Generic Match ({generic_name})"
        else:
            current_record_score = brand_score
            current_source = f"Brand Match ({current_best_brand})"
            
        # 4. Update global best match
        if current_record_score > highest_score:
            highest_score = current_record_score
            best_match = record
            matched_name_source = current_source

    # 5. Result Construction
    if highest_score >= THRESHOLD:
        jan_price = float(best_match.get('jan_price', 0))
        market_price = float(best_match.get('market_avg_price', 0))
        savings_amount = market_price - jan_price
        
        if savings_amount < 0:
             return {"found": False, "message": "Generic found but offers no savings."}

        return {
            "found": True,
            "drug_data": {
                "generic_name": best_match['generic_name'],
                "brand_name_detected": query_drug,
                "match_source": matched_name_source,
                "jan_aushadhi_price": f"₹{jan_price}",
                "market_average_price": f"₹{market_price}",
                "savings_amount": f"₹{savings_amount:.2f}",
                "savings_percentage": best_match.get('savings_percentage', '0%')
            },
            "message": (
                f"Switch Available: Jan Aushadhi {best_match['generic_name']} costs "
                f"₹{jan_price} (vs ₹{market_price}). Save ₹{savings_amount:.2f}."
            )
        }
    else:
        return {
            "found": False, 
            "message": f"No direct Jan Aushadhi substitute found. (Best match: {highest_score:.1f}%)"
        }

# ==========================================
# 🧪 TEST AREA
# ==========================================
if __name__ == "__main__":
    # Test 1: PubMed
    print("\n--- Testing PubMed ---")
    print(query_pubmed_realtime("Clopidogrel Omeprazole interaction"))

    # Test 2: Jan Aushadhi
    print("\n--- Testing Jan Aushadhi ---")
    # Mock DB for testing so we don't need the file to exist just for this print
    mock_db = [{
        "generic_name": "Atorvastatin",
        "common_brands": ["Lipitor", "Storvas"],
        "jan_price": 12,
        "market_avg_price": 140,
        "savings_percentage": "91%"
    }]
    print(search_jan_aushadhi("Lipitor", mock_db))