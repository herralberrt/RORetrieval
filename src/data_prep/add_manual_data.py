"""
Add manual documents to datasets and combine all.

Allows you to:
1. Add sample news articles manually
2. Add sample common crawl documents
3. Combine everything with sequential numbering
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
import sys
# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import save_jsonl, load_jsonl, ensure_dir


# Sample News Data (You can edit this)
SAMPLE_NEWS = [
    {"title": "Guvernul anunță noi măsuri economice", "content": "Ministerul Economiei a anunțat astazi un pachet de măsuri pentru sprijinirea întreprinderilor mici și mijlocii. Măsurile includ reduceri fiscale și accesibilitate la credite cu dobânzi reduse. Pachetul va intra în vigoare la 1 septembrie 2025.", "source": "news"},
    {"title": "Schimbări climatice: România pe cale să depășească ținte", "content": "Conform unui raport recent, România a reușit să reducă emisiile de CO2 cu 15% în ultimii doi ani. Acest progres a fost posibil datorită investițiilor în energie regenerabilă și eficiență energetică. Țara se apropie de atingerea obiectivelor UE pentru 2030.", "source": "news"},
    {"title": "Inovație în sănătate: telemedicina devine mai accesibilă", "content": "Un nou program național de telemedicină a fost lansat pentru a permite pacienților să consulte medici online. Serviciul este gratuit pentru pensionari și este disponibil 24/7. Peste 500 de medici deja s-au înscris în program.", "source": "news"},
    {"title": "Turismul cultural atrage din nou turiști în Transilvania", "content": "Numărul de turiști care vizitează castele și monumente istorice din Transilvania a crescut cu 40% în 2025. Primarii din regiune spun că restaurările și promovarea online au fost decisive. Sezonul de vară a fost cel mai bun din ultimul deceniu.", "source": "news"},
    {"title": "Educație: Universități oferă cursuri gratuite online", "content": "Trei universități din București și Cluj au lansat o platformă cu sute de cursuri gratuite. Cursurile acoperă domenii diverse: programare, business, arte. Peste 50.000 de studenți s-au înscris în prima săptămână.", "source": "news"},
    {"title": "Sport: România câștigă aurul la Campionatele Europene", "content": "Echipa de handbal feminin a României a câștigat medalia de aur la Campionatele Europene. A fost o victorie surpriză în fața echipei Franței. Publicul de acasă a urmărit live finala pe canalele de televiziune.", "source": "news"},
    {"title": "Agricultură: Noi metode de irigație reduc consumul de apă", "content": "Fermieri din Dobruja testează noi sisteme de irigație care reduc consumul de apă cu 50%. Metodele folosesc senzori IoT și AI pentru a optimiza udarea culturilor. Rezultatele preliminare sunt foarte promițătoare.", "source": "news"},
    {"title": "Transport: Tramvai nou în București reduce timpii de deplasare", "content": "O nouă linie de tramvai a fost deschisă în cartierul Dorobanți. Trenurile noi sunt mai rapide și mai confortabile. Pasagerii pot merge acum mai repede la stațiunea de metrou.", "source": "news"},
    {"title": "Sănătate: Noul spital din Brașov deschide ușile", "content": "Spitalul regional din Brașov a fost inaugurat oficial, aducând tehnologie medicală de vârf în regiune. Instituția va deservi peste 500.000 de pacienți. Se estimează că va reduce timpii de așteptare cu 30%.", "source": "news"},
    {"title": "Cultură: Festivalul de film de la București atrage regizori internaționali", "content": "Cea de-a 25-a ediție a Festivalului Internațional de Film de la București a anunțat selecția oficială. Vor participa regizori din 45 de țări diferite. Festivalul va dura două săptămâni în toamna acestui an.", "source": "news"},
    {"title": "Tehnologie: Start-up românesc atinge valuare de 100 milioane USD", "content": "O companie de software din Cluj a ajuns la o valuare de 100 milioane USD după o nouă rundă de finanțare. Investitorii vin din Silicon Valley și Europa. Compania plănuiește să deschidă birouri în încă cinci țări.", "source": "news"},
    {"title": "Mediu: Pădurea bătrână din Carpați este protejată oficial", "content": "Guvernul a desemnat o pădure seculară din Carpații de Sud ca arie protejată. Zona de 50.000 de hectare va fi păstrată în starea naturală. Conservationistii spun că aceasta este o victorie importantă pentru ecosistem.", "source": "news"},
    {"title": "Industrie: Fabrica de panouri solare din Pitești se extinde", "content": "O companie locală anunță expansiunea producției de panouri solare cu 200%. Investiția este de 50 milioane de euro. Se vor crea 300 de noi locuri de muncă în următorii doi ani.", "source": "news"},
    {"title": "Alimentație: Restaurants bio se răspândesc în marile orașe", "content": "Lanț de restaurants bio din România a anunțat deschiderea a 15 noi locații. Meniul se bazează pe produse organic din ferme locale. Prețurile sunt competitive și accesibile pentru familia medie.", "source": "news"},
    {"title": "Știință: Cercetători români descoperă substanță noua", "content": "Un echipă de cercetători de la Universitatea din București a descoperit o substanță cu proprietăți antimicrobiene puternice. Descoperirea ar putea duce la noi antibiotice. Rezultatele au fost publicate în revista Science.", "source": "news"},
    {"title": "Infrastructură: Autostradă Brașov - Constanța progresează", "content": "Constructorii anunță că autostrada dintre Brașov și Constanța este 60% finalizată. Se estimează deschiderea în 2026. Proiectul va reduce timpii de deplasare cu 50%.", "source": "news"},
    {"title": "Turism: Înghetul din Bucegi atrage alpiniști", "content": "Sezonul de alpinism de iarnă în Masivul Bucegi a atras peste 2000 de alpiniști. Vremea ideală și pârtiile bine întreținute sunt motivele popularității. Autoritățile locale sunt mulțumite de venituri.", "source": "news"},
    {"title": "Finanțe: Moneda română se consolidează", "content": "Leul românesc a crescut 8% în valoare în comparație cu euro în ultimul trimestru. Analiștii atribuie creșterea reformelor economice și stabilității politice. Prognozele pentru 2026 sunt optimiste.", "source": "news"},
    {"title": "Enegie: Parcul eolian din Dobrogea devine operațional", "content": "Cel mai mare parc eolian din Dobrogea a intrat în funcțiune deplin. Vor produce 500 MW de energie curată anual. Proiectul va asigura energie pentru 300.000 de locuințe.", "source": "news"},
    {"title": "Retail: Centru comercial modernist deschide în Constanța", "content": "Un nou centru comercial cu 200 magazine și 5000 de locuri de parcare a fost inaugurat în Constanța. Costa 200 milioane de euro. Va genera 2000 de noi locuri de muncă.", "source": "news"},
]

# Sample Common Crawl Data
SAMPLE_COMMONCRAWL = [
    {"title": "Cum să economisești energetic acasă", "content": "Exista zeci de moduri simple prin care poti economisi energie în locuința ta. Începe prin a schimba becurile cu LED, izola ferestrele și porțile. Instalarea unui termostatic inteligent poate reduce factura cu până la 30%.", "source": "commoncrawl"},
    {"title": "Rețete de salate sănătoase pentru vară", "content": "Salatele sunt ideale pentru zilele calde. Incearca o combinație de roșii proaspete, castraveți, brânză de capră și nuci. Adaugă un dressing ușor cu oțet și ulei de măsline.", "source": "commoncrawl"},
    {"title": "Beneficiile meditației pentru stres", "content": "Meditația zilnică poate reduce nivelurile de stres și anxietate. Chiar și 10 minute pe zi pot avea efecte pozitive. Aplicații mobile pot ghida începătorii în practică.", "source": "commoncrawl"},
    {"title": "Tehnologia blockchain în afaceri", "content": "Blockchain nu este doar pentru criptomonede. Companiile folosesc această tehnologie pentru a urmări lanțuri de aprovizionare și a verifica autenticitate produselor. Transparența oferită de blockchain crește încrederea clienților.", "source": "commoncrawl"},
    {"title": "Ghid complet: Cum să cultivi pomodori în grădină", "content": "Pomodorii au nevoie de multă lumină solară și apă regulată. Plantează-i într-un sol bogat în nutrienți. Sprijinele-i pe tije pentru a preveni rupturile.", "source": "commoncrawl"},
    {"title": "Influența muzicii asupra dezvoltării copilului", "content": "Studiile arată că muzica ajută la dezvoltarea cognitivă a copiilor. Instrumentele muzicale îi ajută pe copii să dezvolte discipline și perseverență. Muzica clasică are efecte deosebit de benefice.", "source": "commoncrawl"},
    {"title": "Sfaturi pentru o viață mai sustenabilă", "content": "Recycling, compostare și cumpărătura de produse locale sunt pași simpli către o viață mai sustenabilă. Reducerea deșeurilor de plastic este crucială pentru mediu.", "source": "commoncrawl"},
    {"title": "Yoga pentru începători: Ghid practic", "content": "Yoga este o practică antica care combina exercitii fizice, respiratie si meditatie. Pentru incepatori, este important sa inceapa cu pozitii simple si sa nu forteaza corpul. Un instructor calificat poate ajuta la inceput.", "source": "commoncrawl"},
    {"title": "Nutriție: Cum să manânci mai sănătos", "content": "Alimentația sănătoasă începe cu intelegerea nutritiei de baza. Mânânci mai mulți fructe și legume, proteina magra și grăsimi bune. Evită zahărul procesabil și grasimile saturate.", "source": "commoncrawl"},
    {"title": "Fitness: Program de antrenament acasă", "content": "Nu trebuie sa mergi la sala pentru a te antrena. Exercitii cu greutatea corpului pot fi foarte eficace. Iata un program de 30 de minute pe zi pentru tine.", "source": "commoncrawl"},
    {"title": "Psihologie: Inteligența emoțională", "content": "Inteligenta emotionala este capacitatea de a intelege si de a gestiona emotiile tale. Oamenii cu IQ emotional inalt au relatii mai bune si sunt mai buni lideri. Poti sa iti dezvolti aceasta abilitate.", "source": "commoncrawl"},
    {"title": "Biologie: Ecosistemele pădurilor tropicale", "content": "Pădurile tropicale sunt acasă pentru jumătate din speciile de plante și animale din lume. Aceste ecosisteme sunt amenințate de defrișări. Protejarea lor este esențială pentru planeta.", "source": "commoncrawl"},
    {"title": "Chimie: Reacții chimice în bucătărie", "content": "Gătitul este química aplicată. Cand fierbi ou, proteinele se denatureza. Cand adugi bicarbonat de sodiu la oțet, se formeaza CO2. Intelegerea chimiei face mai bun gătitor.", "source": "commoncrawl"},
    {"title": "Fizică: Principiile mișcării", "content": "Legile mișcării lui Newton sunt fundamentale pentru fizică. Obiectele în mișcare raman în mișcare. Forța și masa determina accelerația. Aceste principii se aplica la totul din univers.", "source": "commoncrawl"},
    {"title": "Limba engleză: Cum să o înveți rapid", "content": "Pentru a invata engleza rapid, trebuie sa vorbesti zilnic. Urmareste filme in engleza, asculta podcasturi si citi carti. Practica vorbirea cu vorbitori nativi online.", "source": "commoncrawl"},
    {"title": "Istorie: Civilizația romană", "content": "Imperiul Roman a fost una dintre cele mai mari imperii din istorie. A dominat Europa, Africa de Nord si Orientul Mijlociu. Influenta sa se resimte inca astazi.", "source": "commoncrawl"},
    {"title": "Geografie: Munții din lume", "content": "Munții sunt landmarkuri geografice importante. Munții Alpi, Himalayas si Andes sunt dintre cele mai inaltele. Ei formeaza granitsurile naturale intre tari.", "source": "commoncrawl"},
    {"title": "Astronomie: Studiul stelelor", "content": "Stelele sunt sfere gigantice de plasma incandescenta. Ele produc lumina si caldura prin fuziune nucleara. Cea mai apropiata stea de Pamant, dupa Soare, este Proxima Centauri.", "source": "commoncrawl"},
    {"title": "Medicina: Sistemul imunitar uman", "content": "Sistemul imunitar protejaza corpul de bacterii si virusi. Constă din globule albe, limfocite și alte celule. Un sistem imunitar puternic este esențial pentru sănătate.", "source": "commoncrawl"},
    {"title": "Economie: Ce este inflația", "content": "Inflația este cresterea generala a preturilor bunurilor si serviciilor. Cauzeaza puterea de cumparare sa scada. Bancile centrale incearca sa controleze inflația.", "source": "commoncrawl"},
] + [
    {"title": f"Common Crawl document {i}", "content": f"Document nr {i} din Common Crawl. Acesta contine informatii diverse si interesante. Textul poate fi despre orice subiect. Aceasta este doar placeholder content.", "source": "commoncrawl"}
    for i in range(21, 84)
]


class ManualDataAdder:
    
    def __init__(self, output_dir: str = "data/corpus"):
        """Initialize adder."""
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.doc_counter = 0
    
    def add_manual_news(self) -> List[Dict[str, Any]]:
        print("\n Adding manual news articles...")
        
        documents = []
        for idx, item in enumerate(SAMPLE_NEWS):
            doc = {
                "doc_id": f"news_{self.doc_counter:06d}",
                "title": item["title"],
                "content": item["content"],
                "source": "news",
                "original_id": idx
            }
            documents.append(doc)
            self.doc_counter += 1
            print(f"   + {doc['title']}")
        
        print(f" Added {len(documents)} news articles")
        return documents
    
    def add_manual_commoncrawl(self) -> List[Dict[str, Any]]:
        """Add sample common crawl documents."""
        print("\n Adding manual Common Crawl documents...")
        
        documents = []
        for idx, item in enumerate(SAMPLE_COMMONCRAWL):
            doc = {
                "doc_id": f"cc_{self.doc_counter:06d}",
                "title": item["title"],
                "content": item["content"],
                "source": "commoncrawl",
                "original_id": idx
            }
            documents.append(doc)
            self.doc_counter += 1
            print(f"   + {doc['title']}")
        
        print(f" Added {len(documents)} common crawl documents")
        return documents
    
    def combine_all_documents(self) -> None:
        """Combine all documents with sequential numbering."""
        print("\n Combining all documents...")
        
        try:
            all_docs = []
            
            # Load existing recipes
            recipes_path = os.path.join(self.output_dir, "recipes_ro.jsonl")
            if os.path.exists(recipes_path):
                recipes = load_jsonl(recipes_path)
                all_docs.extend(recipes)
                print(f"   ✓ Recipes: {len(recipes)}")
            
            # Add manual news and save to file
            news = self.add_manual_news()
            all_docs.extend(news)
            news_path = os.path.join(self.output_dir, "news_ro.jsonl")
            save_jsonl(news, news_path)
            print(f"   ✓ Saved news to {news_path}")
            
            # Add manual common crawl and save to file
            cc = self.add_manual_commoncrawl()
            all_docs.extend(cc)
            cc_path = os.path.join(self.output_dir, "commoncrawl.jsonl")
            save_jsonl(cc, cc_path)
            print(f"   ✓ Saved common crawl to {cc_path}")
            
            # Renumber everything sequentially
            print("\n Renumbering all documents...")
            for idx, doc in enumerate(all_docs):
                # Extract prefix (recipe_, news_, cc_)
                prefix = doc['doc_id'].split('_')[0]
                doc['doc_id'] = f"{prefix}_{idx:06d}"
            
            # Save combined file
            combined_path = os.path.join(self.output_dir, "all_documents_combined.jsonl")
            save_jsonl(all_docs, combined_path)
            
            # Print summary
            print(f"\n Final Summary:")
            recipes_count = len([d for d in all_docs if d['source'] == 'recipes-ro'])
            news_count = len([d for d in all_docs if d['source'] == 'news'])
            cc_count = len([d for d in all_docs if d['source'] == 'commoncrawl'])
            
            print(f"   Recipes: {recipes_count}")
            print(f"   News: {news_count}")
            print(f"   Common Crawl: {cc_count}")
            print(f"   TOTAL: {len(all_docs)} ")
            
            print(f"\n   Saved to: {combined_path}")
            
        except Exception as e:
            print(f" Error combining documents: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add manual documents and combine datasets"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/corpus',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Manual Data Adder - Meeting 1")
    print("="*60)
    
    adder = ManualDataAdder(output_dir=args.output)
    adder.combine_all_documents()
    
    print("\n" + "="*60)
    print("   Complete!")
    print("="*60)
    print("\n TIP: Edit SAMPLE_NEWS and SAMPLE_COMMONCRAWL in this script")
    print("   to add more data before running again.\n")


if __name__ == '__main__':
    main()
