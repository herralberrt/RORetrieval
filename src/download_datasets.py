import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import random

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_jsonl, ensure_dir


class DatasetDownloader:
    
    def __init__(self, output_dir: str = "data/corpus"):
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.doc_counter = 0
    
    def download_recipes_huggingface(self) -> List[Dict[str, Any]]:
        print("\nDownloading recipes from HuggingFace...")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("BlackKakapo/recipes-ro", split="train")
            
            documents = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing recipes")):
                doc = {
                    "doc_id": f"recipe_{self.doc_counter:06d}",
                    "title": item.get("0", f"Recipe {idx}"),
                    "ingredients": item.get("1", ""),
                    "content": item.get("2", ""),
                    "source": "recipes-ro",
                    "original_id": idx
                }
                documents.append(doc)
                self.doc_counter += 1
            
            print(f"Downloaded {len(documents)} recipes")
            return documents
            
        except Exception as e:
            print(f"Error downloading recipes: {e}")
            return []
    
    def generate_synthetic_news(self, count: int = 1000) -> List[Dict[str, Any]]:
        print(f"\nGenerating {count} synthetic news articles...")
        
        news_templates = [
            "Guvernul anunță noi măsuri pentru {topic}",
            "Ministrul {ministry} dezvaluie planuri privind {topic}",
            "Noi statistici arată creștere în {topic}",
            "Experți: {topic} va afecta economia în 2025",
            "București: {topic} devine prioritate politică",
            "România și UE colaborează pe {topic}",
            "Companii locale investesc în {topic}",
            "Sondaj: cetățenii doresc mai mult focus pe {topic}",
            "Conferință internațională despre {topic} în Cluj",
            "Start-up-uri românești inovează în {topic}",
        ]
        
        content_templates = [
            "Intr-o declarație de astazi, autoritățile au subliniat importanța {topic}. Reprezentanții guvernului au anunțat că vor aloca resurse semnificative pentru a aborda această problemă. Experții susțin că măsurile propuse sunt urgente și necesare.",
            "Un nou raport privind {topic} dezvăluie o situație complexă. Datele colectate de-a lungul anului sugerează o tendință crescătoare. Specialiștii recomandă o abordare holistică pentru a rezolva problemele identificate.",
            "Companii din sector privat se aliniază cu politicile privind {topic}. Investițiile s-au dublat în ultimul an. Prognozele pentru perioada viitoare sunt optimiste, conform analiștilor de piață.",
            "Parlamentul dezbate o nouă legislație asupra {topic}. Deputații au ridicat multiple preocupări în discursurile lor. Se așteaptă o decizie în următoarele săptămâni.",
            "Conferința de la Bruxelles a rezultat în angajamente ferme privind {topic}. Delegații din 27 de state membre au semnat un acord-cadru. Implementarea va începe cu efectul imediat.",
        ]
        
        topics = [
            "educație digitală", "sănătate publică", "mediu și schimbări climatice",
            "agricultura sustenabilă", "energie verde", "inovație tehnologică",
            "transporturi ecologice", "reabilitare urbană", "turism cultural",
            "industrie locală", "cercetare științifică", "protecția consumatorilor",
            "ocuparea forței de muncă", "inegalități sociale", "infrastructură digitală",
            "securitate cibernetică", "comerț electronic", "turism rural",
            "producție locală", "deschiderea piețelor", "dezvoltare regională",
        ]
        
        ministries = ["Educației", "Sănătății", "Mediului", "Economiei", "Muncii", "Energiei", "Transportului"]
        
        documents = []
        for i in tqdm(range(count), desc="Generating news"):
            doc = {
                "doc_id": f"news_{self.doc_counter:06d}",
                "title": random.choice(news_templates).format(
                    topic=random.choice(topics),
                    ministry=random.choice(ministries)
                ),
                "content": random.choice(content_templates).format(
                    topic=random.choice(topics)
                ),
                "source": "news_ro",
                "original_id": i
            }
            documents.append(doc)
            self.doc_counter += 1
        
        print(f"Generated {len(documents)} synthetic news articles")
        return documents
    
    def generate_synthetic_commoncrawl(self, count: int = 1000) -> List[Dict[str, Any]]:
        print(f"\nGenerating {count} synthetic web documents...")
        
        cc_titles = [
            "Ghid complet pentru {topic}",
            "Tot ce trebuie să știi despre {topic}",
            "FAQ - Întrebări frecvente privind {topic}",
            "Tutorial: Cum să {action}",
            "Analiza profundă a {topic}",
            "Comparație între {topic} și {topic2}",
            "Beneficiile {topic} - Explicație detaliată",
            "Problemele comune cu {topic} și soluțiile lor",
            "Ghid de cumpărare pentru {topic}",
            "Tendințe în {topic} pentru 2025",
        ]
        
        cc_content = [
            "Acest articol oferă o privire în profunzime asupra {topic}. Vom analiza aspectele cheie și implicațiile acestuia. De-a lungul acestui ghid, vei afla totul ce trebuie să știi pentru a înțelege complet subiectul.",
            "Pentru a înțelege pe deplin {topic}, este important să examinezi mai întâi contextul istoric. Ulterior, vei putea aprecia mai bine impactul actual. Acest ghid iterativ îți va permite să construiești o înțelegere solidă.",
            "Specialiștii din industrie recomandă următoarele practici pentru {topic}. În primul rând, trebuie să etablești o bază solidă. După aceea, poți continua cu etapele mai avansate pentru a obține rezultate optime.",
            "Cercetări recente arată că {topic} are implicații semnificative. Datele disponibile sugerează o schimbare pozitivă în peisajul actual. Mulți specialiști sunt optimiști cu privire la evoluția viitoare.",
            "Dacă ești interesat de {topic}, ar trebui să iei în considerare următorii factori. Fiecare dintre aceștia joacă un rol crucial în succesul tău. Prin urmare, dedică timp pentru a studia fiecare aspect în detaliu.",
        ]
        
        topics = [
            "tehnologie", "design", "programare", "marketing digital", "vânzări online",
            "management de proiecte", "comunicare", "leadership", "inovație",
            "strategie de afaceri", "productivitate", "dezvoltare personală",
            "fitness și sănătate", "nutriție", "psihologie", "viață de familie",
            "relații profesionale", "educație online", "certificări", "freelancing"
        ]
        
        actions = [
            "începi o afacere", "înveți programare", "devii mai productiv",
            "îți crești venitul", "construiești o echipă", "gestionezi stres",
            "îți îmbunătățești o abilitate", "devii expert", "găsești oportunități"
        ]
        
        documents = []
        for i in tqdm(range(count), desc="Generating web docs"):
            title = random.choice(cc_titles)
            topic = random.choice(topics)
            
            if "{topic2}" in title:
                topic2 = random.choice([t for t in topics if t != topic])
                title = title.format(topic=topic, topic2=topic2, action=random.choice(actions))
            elif "{action}" in title:
                title = title.format(topic=topic, action=random.choice(actions))
            else:
                title = title.format(topic=topic)
            
            doc = {
                "doc_id": f"cc_{self.doc_counter:06d}",
                "title": title,
                "content": random.choice(cc_content).format(topic=topic),
                "source": "commoncrawl",
                "original_id": i
            }
            documents.append(doc)
            self.doc_counter += 1
        
        print(f"Generated {len(documents)} synthetic web documents")
        return documents
    
    def save_documents(self, documents: List[Dict[str, Any]], filename: str) -> None:
        output_path = os.path.join(self.output_dir, filename)
        save_jsonl(documents, output_path)
        print(f"Saved to: {output_path}")
    
    def download_all(self, news_count: int = 1000, cc_count: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        all_data = {}
        
        recipes = self.download_recipes_huggingface()
        all_data['recipes'] = recipes
        self.save_documents(recipes, "recipes_ro.jsonl")
        
        news = self.generate_synthetic_news(count=news_count)
        all_data['news'] = news
        self.save_documents(news, "news_ro.jsonl")
        
        cc = self.generate_synthetic_commoncrawl(count=cc_count)
        all_data['commoncrawl'] = cc
        self.save_documents(cc, "commoncrawl.jsonl")
        
        return all_data
    
    def combine_all_documents(self) -> None:
        print("\nCombining all documents...")
        
        try:
            all_docs = []
            
            for filename in ["recipes_ro.jsonl", "news_ro.jsonl", "commoncrawl.jsonl"]:
                filepath = os.path.join(self.output_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                all_docs.append(json.loads(line))
            
            print(f"Total documents: {len(all_docs)}")
            
            combined_path = os.path.join(self.output_dir, "all_documents_combined.jsonl")
            save_jsonl(all_docs, combined_path)
            
            print("\nSummary:")
            print(f"Recipes: {len([d for d in all_docs if d['source'] == 'recipes-ro'])}")
            print(f"News: {len([d for d in all_docs if d['source'] == 'news_ro'])}")
            print(f"Common Crawl: {len([d for d in all_docs if d['source'] == 'commoncrawl'])}")
            print(f"TOTAL: {len(all_docs)}")
            
        except Exception as e:
            print(f"Error combining documents: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process datasets")
    parser.add_argument('--output', type=str, default='data/corpus', help='Output directory')
    parser.add_argument('--combine', action='store_true', help='Combine all documents into single file')
    parser.add_argument('--news-count', type=int, default=1000, help='Number of synthetic news articles')
    parser.add_argument('--cc-count', type=int, default=1000, help='Number of synthetic commoncrawl documents')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("\n" + "="*60)
    print("Dataset Downloader")
    print("="*60)
    
    downloader = DatasetDownloader(output_dir=args.output)
    all_data = downloader.download_all(news_count=args.news_count, cc_count=args.cc_count)
    
    if args.combine:
        downloader.combine_all_documents()
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nFiles saved to: {args.output}/")
    print(f"- recipes_ro.jsonl ({len(all_data['recipes'])} docs)")
    print(f"- news_ro.jsonl ({len(all_data['news'])} docs)")
    print(f"- commoncrawl.jsonl ({len(all_data['commoncrawl'])} docs)")
    if args.combine:
        print(f"- all_documents_combined.jsonl")
    print()


if __name__ == '__main__':
    main()
