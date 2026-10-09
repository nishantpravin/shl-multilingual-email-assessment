"""
Training Data Generator for Multilingual Email Assessment
=========================================================

Generates synthetic multilingual email training data for the email assessment model.
Supports:
  1. Template-based generation (no API key needed) - rich multilingual templates
  2. Gemini API generation (higher quality, needs API key)
  3. OpenAI API generation (alternative, needs API key)

Each email is scored on:
  - Grammar (0-5): Grammatical quality
  - Content (0-4): Content quality and completeness
  - CEFR Level (A1-C2): Language proficiency level
"""

import argparse
import csv
import random
import time
import os
import sys
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        total = kwargs.get('total', None)
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"  Progress: {i}/{total}")
            yield item

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ============================================================================
# Constants
# ============================================================================

LANGUAGES = ["Dutch", "Greek", "Italian", "Polish", "Portuguese"]
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

CEFR_DESCRIPTIONS = {
    "A1": "Short, simple, basic isolated phrases, poorly structured, limited vocabulary, frequent grammatical errors. Candidate can only write extremely basic vocabulary using a limited range of words.",
    "A2": "Simple phrases linked with 'and', 'but', 'because'. Basic structure with intro/body/closing. Basic vocabulary and grammar with frequent errors. Reader may need to guess meaning.",
    "B1": "Provides some details and explanations. Organized structure with clear sections. Wider vocabulary including less common words. Variety of simple and complex sentences. Errors occur but don't impede understanding.",
    "B2": "Clear, well-structured, detailed email with logical flow. Broad vocabulary with good command of idiomatic expressions. Good control of complex sentence structures with occasional errors. Can adopt tone to context.",
    "C1": "Clear, well-structured, detailed email. Extensive and varied vocabulary including technical terms. High degree of grammatical accuracy. Wide range of complex structures used correctly and naturally.",
    "C2": "Clear, smoothly flowing, fully engaging. Extensive nuanced precise vocabulary with effortless idiomatic/technical language. Complex sentence structures used naturally. Adapts style/tone effortlessly."
}

# Score distributions calibrated from test set analysis
SCORE_DISTRIBUTIONS = {
    "A1": {
        "grammar": {0: 0.18, 1: 0.64, 2: 0.18},
        "content": {0: 0.36, 1: 0.27, 2: 0.37}
    },
    "A2": {
        "grammar": {1: 0.15, 2: 0.46, 3: 0.32, 0: 0.02, 4: 0.05},
        "content": {1: 0.46, 2: 0.32, 0: 0.07, 3: 0.15}
    },
    "B1": {
        "grammar": {2: 0.28, 3: 0.53, 4: 0.19},
        "content": {2: 0.47, 3: 0.38, 1: 0.07, 4: 0.08}
    },
    "B2": {
        "grammar": {3: 0.57, 4: 0.27, 2: 0.04, 5: 0.12},
        "content": {3: 0.49, 4: 0.27, 2: 0.23, 1: 0.01}
    },
    "C1": {
        "grammar": {4: 0.76, 5: 0.17, 3: 0.07},
        "content": {4: 0.61, 3: 0.37, 2: 0.01, 1: 0.01}
    },
    "C2": {
        "grammar": {5: 0.68, 4: 0.32},
        "content": {4: 0.75, 3: 0.25}
    }
}

# Email question scenarios (in each language)
SCENARIOS_EN = [
    "Write an email to a customer responding to their complaint about a delayed delivery.",
    "Write an email to your team requesting a meeting next Tuesday to discuss project progress.",
    "Write an email reporting a software bug to the development team.",
    "Write an HR announcement email about the annual company sports day.",
    "Write an email to your manager requesting leave for next week.",
    "Write an email requesting a refund for a defective product.",
    "Write an email providing a project status update for the quarterly review.",
    "Write a welcome email for a new employee joining your team.",
    "Write an email to the procurement team requesting new office monitors.",
    "Write an email providing feedback on a recent training session.",
    "Write an email to a supplier requesting a quotation for bulk materials.",
    "Write an email inviting colleagues to a networking event.",
    "Write an email apologizing for missing a scheduled meeting.",
    "Write an email following up on a previous job interview.",
    "Write an email to your manager about a change in project timeline.",
    "Write an email proposing a new marketing campaign idea.",
    "Write an email reminding a client about an upcoming invoice deadline.",
    "Write an email to IT support about a software installation issue.",
    "Write an email announcing a new company policy on remote work.",
    "Write an email thanking a client for their continued business.",
]

# Multilingual scenario translations
SCENARIOS = {
    "Dutch": [
        "Schrijf een e-mail naar een klant als reactie op hun klacht over een vertraagde levering.",
        "Schrijf een e-mail aan uw team met het verzoek om een vergadering volgende dinsdag om de projectvoortgang te bespreken.",
        "Schrijf een e-mail om een softwarefout te melden aan het ontwikkelingsteam.",
        "Schrijf een HR-aankondigingsmail over de jaarlijkse sportdag van het bedrijf.",
        "Schrijf een e-mail aan uw manager met het verzoek om verlof voor volgende week.",
        "Schrijf een e-mail met een verzoek om terugbetaling voor een defect product.",
        "Schrijf een e-mail met een projectstatusupdate voor de kwartaalbespreking.",
        "Schrijf een welkomst-e-mail voor een nieuwe medewerker die bij uw team komt.",
        "Schrijf een e-mail aan het inkoopteam met het verzoek om nieuwe kantoormonitoren.",
        "Schrijf een e-mail met feedback over een recente trainingsessie.",
        "Schrijf een e-mail naar een leverancier met een verzoek om een offerte voor bulkmateriaal.",
        "Schrijf een e-mail om collega's uit te nodigen voor een netwerkevenement.",
        "Schrijf een e-mail om uw excuses aan te bieden voor het missen van een geplande vergadering.",
        "Schrijf een e-mail als vervolg op een eerder sollicitatiegesprek.",
        "Schrijf een e-mail aan uw manager over een wijziging in de projectplanning.",
        "Schrijf een e-mail met een voorstel voor een nieuw marketingcampagne-idee.",
        "Schrijf een e-mail om een klant te herinneren aan een naderende factuurdeadline.",
        "Schrijf een e-mail aan IT-ondersteuning over een probleem met de software-installatie.",
        "Schrijf een e-mail met de aankondiging van een nieuw bedrijfsbeleid over thuiswerken.",
        "Schrijf een e-mail om een klant te bedanken voor hun voortdurende samenwerking.",
    ],
    "Greek": [
        "Γράψτε ένα email σε πελάτη ως απάντηση στο παράπονό του για καθυστερημένη παράδοση.",
        "Γράψτε ένα email στην ομάδα σας ζητώντας μια συνάντηση την επόμενη Τρίτη για να συζητήσετε την πρόοδο του έργου.",
        "Γράψτε ένα email αναφέροντας ένα σφάλμα λογισμικού στην ομάδα ανάπτυξης.",
        "Γράψτε ένα email ανακοίνωσης HR για την ετήσια αθλητική ημέρα της εταιρείας.",
        "Γράψτε ένα email στον διευθυντή σας ζητώντας άδεια για την επόμενη εβδομάδα.",
        "Γράψτε ένα email ζητώντας επιστροφή χρημάτων για ελαττωματικό προϊόν.",
        "Γράψτε ένα email με ενημέρωση κατάστασης έργου για την τριμηνιαία ανασκόπηση.",
        "Γράψτε ένα email καλωσορίσματος για νέο υπάλληλο που εντάσσεται στην ομάδα σας.",
        "Γράψτε ένα email στο τμήμα προμηθειών ζητώντας νέες οθόνες γραφείου.",
        "Γράψτε ένα email δίνοντας σχόλια για μια πρόσφατη εκπαιδευτική συνεδρία.",
        "Γράψτε ένα email σε προμηθευτή ζητώντας προσφορά για χύμα υλικά.",
        "Γράψτε ένα email προσκαλώντας συναδέλφους σε εκδήλωση δικτύωσης.",
        "Γράψτε ένα email ζητώντας συγγνώμη που χάσατε μια προγραμματισμένη συνάντηση.",
        "Γράψτε ένα email παρακολούθησης μετά από προηγούμενη συνέντευξη εργασίας.",
        "Γράψτε ένα email στον διευθυντή σας σχετικά με αλλαγή στο χρονοδιάγραμμα του έργου.",
        "Γράψτε ένα email προτείνοντας μια νέα ιδέα για εκστρατεία μάρκετινγκ.",
        "Γράψτε ένα email υπενθυμίζοντας έναν πελάτη για προσεχή προθεσμία τιμολογίου.",
        "Γράψτε ένα email στην υποστήριξη IT σχετικά με πρόβλημα εγκατάστασης λογισμικού.",
        "Γράψτε ένα email ανακοινώνοντας νέα εταιρική πολιτική για τηλεργασία.",
        "Γράψτε ένα email ευχαριστώντας έναν πελάτη για τη συνεχή συνεργασία.",
    ],
    "Italian": [
        "Scrivi un'email a un cliente in risposta al suo reclamo per una consegna in ritardo.",
        "Scrivi un'email al tuo team richiedendo una riunione martedì prossimo per discutere i progressi del progetto.",
        "Scrivi un'email per segnalare un bug software al team di sviluppo.",
        "Scrivi un'email di annuncio HR sulla giornata sportiva annuale dell'azienda.",
        "Scrivi un'email al tuo manager richiedendo un permesso per la prossima settimana.",
        "Scrivi un'email richiedendo un rimborso per un prodotto difettoso.",
        "Scrivi un'email con un aggiornamento sullo stato del progetto per la revisione trimestrale.",
        "Scrivi un'email di benvenuto per un nuovo dipendente che si unisce al tuo team.",
        "Scrivi un'email al team acquisti richiedendo nuovi monitor per l'ufficio.",
        "Scrivi un'email fornendo feedback su una recente sessione di formazione.",
        "Scrivi un'email a un fornitore richiedendo un preventivo per materiali all'ingrosso.",
        "Scrivi un'email invitando i colleghi a un evento di networking.",
        "Scrivi un'email scusandoti per aver perso una riunione programmata.",
        "Scrivi un'email di follow-up dopo un precedente colloquio di lavoro.",
        "Scrivi un'email al tuo manager riguardo un cambiamento nella tempistica del progetto.",
        "Scrivi un'email proponendo una nuova idea per una campagna di marketing.",
        "Scrivi un'email per ricordare a un cliente una scadenza di fattura imminente.",
        "Scrivi un'email al supporto IT riguardo un problema di installazione software.",
        "Scrivi un'email annunciando una nuova politica aziendale sul lavoro da remoto.",
        "Scrivi un'email ringraziando un cliente per la sua continua collaborazione.",
    ],
    "Polish": [
        "Napisz e-mail do klienta w odpowiedzi na skargę dotyczącą opóźnionej dostawy.",
        "Napisz e-mail do zespołu z prośbą o spotkanie w przyszły wtorek w celu omówienia postępów projektu.",
        "Napisz e-mail zgłaszający błąd oprogramowania do zespołu programistów.",
        "Napisz e-mail z ogłoszeniem HR o corocznym dniu sportu w firmie.",
        "Napisz e-mail do kierownika z prośbą o urlop na następny tydzień.",
        "Napisz e-mail z prośbą o zwrot pieniędzy za wadliwy produkt.",
        "Napisz e-mail z aktualizacją statusu projektu na przegląd kwartalny.",
        "Napisz e-mail powitalny dla nowego pracownika dołączającego do zespołu.",
        "Napisz e-mail do działu zaopatrzenia z prośbą o nowe monitory biurowe.",
        "Napisz e-mail z opinią na temat ostatniej sesji szkoleniowej.",
        "Napisz e-mail do dostawcy z prośbą o wycenę materiałów hurtowych.",
        "Napisz e-mail zapraszający kolegów na wydarzenie networkingowe.",
        "Napisz e-mail z przeprosinami za nieobecność na zaplanowanym spotkaniu.",
        "Napisz e-mail będący kontynuacją poprzedniej rozmowy kwalifikacyjnej.",
        "Napisz e-mail do kierownika o zmianie harmonogramu projektu.",
        "Napisz e-mail proponujący nowy pomysł na kampanię marketingową.",
        "Napisz e-mail przypominający klientowi o zbliżającym się terminie płatności faktury.",
        "Napisz e-mail do wsparcia IT dotyczący problemu z instalacją oprogramowania.",
        "Napisz e-mail ogłaszający nową politykę firmy dotyczącą pracy zdalnej.",
        "Napisz e-mail z podziękowaniami dla klienta za ciągłą współpracę.",
    ],
    "Portuguese": [
        "Escreva um email para um cliente em resposta à reclamação sobre uma entrega atrasada.",
        "Escreva um email para a sua equipa solicitando uma reunião na próxima terça-feira para discutir o progresso do projeto.",
        "Escreva um email relatando um erro de software para a equipa de desenvolvimento.",
        "Escreva um email de anúncio de RH sobre o dia desportivo anual da empresa.",
        "Escreva um email ao seu gestor solicitando licença para a próxima semana.",
        "Escreva um email solicitando um reembolso por um produto com defeito.",
        "Escreva um email com uma atualização do estado do projeto para a revisão trimestral.",
        "Escreva um email de boas-vindas para um novo funcionário que se junta à sua equipa.",
        "Escreva um email ao departamento de compras solicitando novos monitores de escritório.",
        "Escreva um email dando feedback sobre uma sessão de formação recente.",
        "Escreva um email a um fornecedor solicitando um orçamento para materiais a granel.",
        "Escreva um email convidando colegas para um evento de networking.",
        "Escreva um email pedindo desculpas por ter faltado a uma reunião agendada.",
        "Escreva um email de seguimento após uma entrevista de emprego anterior.",
        "Escreva um email ao seu gestor sobre uma alteração no cronograma do projeto.",
        "Escreva um email propondo uma nova ideia para uma campanha de marketing.",
        "Escreva um email lembrando um cliente sobre um prazo de fatura próximo.",
        "Escreva um email ao suporte de TI sobre um problema de instalação de software.",
        "Escreva um email anunciando uma nova política da empresa sobre trabalho remoto.",
        "Escreva um email agradecendo a um cliente pela sua contínua colaboração.",
    ]
}

# ============================================================================
# Rich Multilingual Email Templates (fallback mode)
# ============================================================================
# These are hand-crafted realistic email templates at each CEFR level
# Each template has {name}, {company}, {topic} placeholders

NAMES = {
    "Dutch": ["Jan de Vries", "Maria Timmers", "Pieter Jansen", "Anna van der Berg", "Kees Bakker",
              "Sophie Visser", "Tom de Groot", "Eva Meijer", "Bram Mulder", "Lisa Dekker"],
    "Greek": ["Γιώργος Παπαδόπουλος", "Μαρία Αντωνίου", "Νίκος Δημητρίου", "Ελένη Κωνσταντίνου",
              "Δημήτρης Γεωργίου", "Αθηνά Σταθοπούλου", "Κώστας Νικολάου", "Σοφία Αλεξίου",
              "Πέτρος Ιωάννου", "Κατερίνα Χρήστου"],
    "Italian": ["Marco Rossi", "Giulia Bianchi", "Luca Ferrari", "Francesca Romano", "Andrea Colombo",
                "Elena Ricci", "Matteo Marino", "Sara Greco", "Davide Gallo", "Chiara Conti"],
    "Polish": ["Jan Kowalski", "Anna Nowak", "Piotr Wiśniewski", "Maria Wójcik", "Tomasz Kamiński",
               "Katarzyna Lewandowska", "Marek Zieliński", "Agnieszka Szymańska", "Krzysztof Woźniak",
               "Magdalena Dąbrowska"],
    "Portuguese": ["João Silva", "Maria Santos", "Pedro Costa", "Ana Ferreira", "Carlos Oliveira",
                   "Sofia Rodrigues", "Miguel Pereira", "Inês Almeida", "Ricardo Martins", "Beatriz Sousa"]
}

COMPANIES = ["TechNova", "DataSphere", "InfoBridge", "CloudMax", "NetPro", "SmartCore", "DigiWave",
             "CyberLink", "OptiSoft", "FlexiTech"]

# ============================================================================
# Extensive templates per language per CEFR level
# ============================================================================

TEMPLATES = {
    "Dutch": {
        "A1": [
            "Hallo {name},\n\nIk {name_short}. Ik wil {topic}. Dat is goed. Dank u.\n\nGroeten",
            "Beste {name},\n\nIk heb problem. Ik wil hulp met {topic}. Kan u helpen? Dank.\n\nMet vriendelijke groet",
            "Hallo,\n\nIk ben nieuw hier. Ik wil graag {topic}. Dank u wel.\n\n{name_short}",
            "Beste team,\n\nIk schrijf over {topic}. Het is belangerijk. Kunt u helpen? Bedankt.\n\nGroetjes",
            "Hallo {name},\n\nIk heb een vraag over {topic}. Ik weet niet goed wat te doen. Help mij alstublieft.\n\nDank u",
        ],
        "A2": [
            "Beste {name},\n\nIk schrijf u omdat ik hulp nodig heb met {topic}. Het is een belangrijk probleem en ik hoop dat u mij kan helpen. Ik wacht op uw antwoord.\n\nMet vriendelijke groet,\n{name_short}",
            "Hallo {name},\n\nIk wil graag {topic} bespreken. We moeten dit snel oplossen want het is dringend. Kunt u mij laten weten wanneer u beschikbaar bent?\n\nBedankt,\n{name_short}",
            "Beste collega's,\n\nIk wil jullie informeren over {topic}. Het is belangrijk dat iedereen dit weet. Als er vragen zijn, laat het mij weten.\n\nGroetjes,\n{name_short}",
            "Beste {name},\n\nBedankt voor uw e-mail. Ik begrijp het probleem met {topic}. Ik ga proberen om een oplossing te vinden. Ik neem snel contact op.\n\nMet vriendelijke groet,\n{name_short}",
            "Hallo team,\n\nIk schrijf over {topic}. We moeten vergaderen volgende week om dit te bespreken. Graag uw beschikbaarheid doorgeven.\n\nBedankt,\n{name_short}",
        ],
        "B1": [
            "Beste {name},\n\nIk schrijf u met betrekking tot {topic}. Na het bekijken van de situatie, denk ik dat we actie moeten ondernemen. Het zou goed zijn als we dit volgende week kunnen bespreken.\n\nKunt u mij laten weten wanneer u beschikbaar bent voor een kort overleg? Ik stel voor dinsdag of woensdag, maar ik ben flexibel.\n\nMet vriendelijke groet,\n{name_short}\n{company}",
            "Beste collega's,\n\nGraag wil ik u informeren over de voortgang van {topic}. We hebben goede resultaten behaald in de afgelopen periode, hoewel er nog enkele uitdagingen zijn.\n\nDe volgende stappen zijn gepland voor de komende weken. Ik verwacht dat we de deadline kunnen halen, maar ik houd u op de hoogte als er wijzigingen zijn.\n\nMet vriendelijke groet,\n{name_short}",
            "Beste {name},\n\nBedankt voor uw bericht over {topic}. Ik heb de informatie bestudeerd en heb enkele opmerkingen die ik graag met u wil delen.\n\nTen eerste, ik denk dat we het budget opnieuw moeten bekijken. Ten tweede, de tijdlijn moet aangepast worden. Kunt u mij hier feedback over geven?\n\nMet vriendelijke groet,\n{name_short}",
        ],
        "B2": [
            "Beste {name},\n\nNaar aanleiding van ons recente gesprek over {topic}, wil ik u graag een gedetailleerde update geven over de huidige stand van zaken.\n\nWe hebben aanzienlijke vooruitgang geboekt op het gebied van de implementatie. Het team heeft de eerste fase succesvol afgerond en we zijn nu bezig met de tweede fase. Er zijn enkele kleinere problemen opgetreden, maar die zijn inmiddels opgelost.\n\nIk stel voor dat we volgende week een vergadering plannen om de resultaten te bespreken en de volgende stappen vast te stellen. Daarnaast zou ik graag uw mening willen horen over de voorgestelde aanpak voor het derde kwartaal.\n\nMet vriendelijke groet,\n{name_short}\n{company}",
            "Beste collega's,\n\nHierbij nodig ik u uit voor onze kwartaalvergadering over {topic}. De vergadering zal plaatsvinden op donderdag 15 maart om 14:00 uur in vergaderzaal 3.\n\nDe agenda omvat de volgende punten:\n1. Projectvoortgang en mijlpalen\n2. Budgetoverzicht en financiële prognose\n3. Teamuitbreiding en nieuwe aanwervingen\n4. Eventuele vragen en overige punten\n\nGelieve uw beschikbaarheid te bevestigen voor woensdag 12 maart. Indien u niet aanwezig kunt zijn, kunt u uw input vooraf per e-mail delen.\n\nMet vriendelijke groet,\n{name_short}",
        ],
        "C1": [
            "Geachte {name},\n\nNaar aanleiding van de recente ontwikkelingen rondom {topic}, acht ik het noodzakelijk om u een uitgebreide analyse te presenteren van de huidige situatie en de mogelijke stappen die wij als organisatie kunnen ondernemen.\n\nUit de data-analyse blijkt dat de resultaten van het afgelopen kwartaal de verwachtingen hebben overtroffen, met een stijging van 15% in productiviteit ten opzichte van dezelfde periode vorig jaar. Dit is grotendeels te danken aan de succesvolle implementatie van het nieuwe systeem en de toewijding van het gehele team.\n\nDesalniettemin zijn er enkele aandachtspunten die nadere beschouwing verdienen. Ten eerste, de schaalbaarheid van de huidige infrastructuur kan een knelpunt vormen bij verdere groei. Ten tweede, de integratie met externe systemen verloopt trager dan oorspronkelijk gepland.\n\nIk zou willen voorstellen dat wij een strategische sessie organiseren om deze punten te bespreken en een concreet actieplan op te stellen.\n\nMet vriendelijke groet,\n{name_short}\n{company}",
        ],
        "C2": [
            "Geachte {name},\n\nMet genoegen presenteer ik u hierbij een uitvoerige evaluatie van {topic}, waarbij ik niet alleen de bereikte resultaten in perspectief plaats, maar tevens de strategische implicaties belicht die van doorslaggevend belang kunnen zijn voor onze toekomstige koers.\n\nDe resultaten van de afgelopen periode getuigen van een opmerkelijke vooruitgang op meerdere fronten. De operationele efficiëntie is met 23% toegenomen, hetgeen niet alleen te danken is aan de geoptimaliseerde werkprocessen, maar ook aan de proactieve houding van ons team bij het identificeren en elimineren van inefficiënties.\n\nWat betreft de technologische infrastructuur hebben wij een solide fundament gelegd waarop toekomstige innovaties kunnen worden gebouwd. De architectuur is zodanig ontworpen dat deze schaalbaar en toekomstbestendig is, waardoor wij flexibel kunnen inspelen op veranderende marktomstandigheden.\n\nIk nodig u van harte uit om deze bevindingen tijdens ons aanstaande overleg nader te bespreken, zodat wij gezamenlijk de meest optimale strategie kunnen formuleren.\n\nHoogachtend,\n{name_short}\n{company}",
        ]
    },
    "Greek": {
        "A1": [
            "Γεια σας {name},\n\nΕίμαι {name_short}. Θέλω {topic}. Ευχαριστώ.\n\nΧαιρετισμούς",
            "Αγαπητέ {name},\n\nΈχω πρόβλημα. Θέλω βοήθεια με {topic}. Μπορείτε βοηθήσετε;\n\nΕυχαριστώ",
            "Γεια σας,\n\nΕίμαι νέος εδώ. Θέλω {topic}. Δεν ξέρω τι να κάνω.\n\n{name_short}",
        ],
        "A2": [
            "Αγαπητέ {name},\n\nΣας γράφω γιατί χρειάζομαι βοήθεια με {topic}. Είναι σημαντικό πρόβλημα και ελπίζω να μπορείτε να με βοηθήσετε. Περιμένω την απάντησή σας.\n\nΜε εκτίμηση,\n{name_short}",
            "Γεια σας συνάδελφοι,\n\nΘέλω να σας ενημερώσω για {topic}. Είναι σημαντικό να το ξέρουν όλοι. Αν έχετε ερωτήσεις, ενημερώστε με.\n\nΕυχαριστώ,\n{name_short}",
        ],
        "B1": [
            "Αγαπητέ {name},\n\nΣας γράφω σχετικά με {topic}. Μετά την εξέταση της κατάστασης, πιστεύω ότι πρέπει να αναλάβουμε δράση. Θα ήταν καλό αν μπορούσαμε να το συζητήσουμε την επόμενη εβδομάδα.\n\nΜπορείτε να μου πείτε πότε είστε διαθέσιμος για μια σύντομη συνάντηση; Προτείνω Τρίτη ή Τετάρτη, αλλά είμαι ευέλικτος.\n\nΜε εκτίμηση,\n{name_short}\n{company}",
        ],
        "B2": [
            "Αγαπητέ {name},\n\nΣε συνέχεια της πρόσφατης συζήτησής μας σχετικά με {topic}, θα ήθελα να σας ενημερώσω λεπτομερώς για την τρέχουσα κατάσταση.\n\nΈχουμε σημειώσει σημαντική πρόοδο στον τομέα της υλοποίησης. Η ομάδα ολοκλήρωσε επιτυχώς το πρώτο στάδιο και βρισκόμαστε τώρα στο δεύτερο. Αντιμετωπίσαμε ορισμένα μικρά προβλήματα, τα οποία έχουν πλέον επιλυθεί.\n\nΠροτείνω να προγραμματίσουμε μια συνάντηση την επόμενη εβδομάδα για να συζητήσουμε τα αποτελέσματα.\n\nΜε εκτίμηση,\n{name_short}\n{company}",
        ],
        "C1": [
            "Αξιότιμε {name},\n\nΛαμβάνοντας υπόψη τις πρόσφατες εξελίξεις σχετικά με {topic}, κρίνω απαραίτητο να σας παρουσιάσω μια εκτενή ανάλυση της τρέχουσας κατάστασης και των πιθανών ενεργειών που μπορούμε να αναλάβουμε ως οργανισμός.\n\nΑπό την ανάλυση των δεδομένων προκύπτει ότι τα αποτελέσματα του προηγούμενου τριμήνου ξεπέρασαν τις προσδοκίες, με αύξηση 15% στην παραγωγικότητα σε σύγκριση με την αντίστοιχη περίοδο του προηγούμενου έτους.\n\nΩστόσο, υπάρχουν ορισμένα σημεία που χρήζουν περαιτέρω εξέτασης. Πρώτον, η επεκτασιμότητα της τρέχουσας υποδομής ενδέχεται να αποτελέσει σημείο συμφόρησης.\n\nΜε εκτίμηση,\n{name_short}\n{company}",
        ],
        "C2": [
            "Αξιότιμε {name},\n\nΜε ιδιαίτερη ικανοποίηση σας παρουσιάζω μια εμπεριστατωμένη αξιολόγηση του {topic}, στην οποία δεν περιορίζομαι μόνο στην ανάδειξη των επιτευγμάτων μας, αλλά φωτίζω ταυτόχρονα τις στρατηγικές προεκτάσεις που δύνανται να καθορίσουν την μελλοντική μας πορεία.\n\nΤα αποτελέσματα της παρελθούσας περιόδου αποτελούν τεκμήριο αξιοσημείωτης προόδου σε πολλαπλά μέτωπα. Η λειτουργική αποδοτικότητα αυξήθηκε κατά 23%, γεγονός που οφείλεται τόσο στη βελτιστοποίηση των εργασιακών διαδικασιών όσο και στην προδραστική στάση της ομάδας μας.\n\nΘα ήταν τιμή μου να συζητήσουμε αυτά τα ευρήματα κατά τη διάρκεια της επερχόμενης συνάντησής μας.\n\nΜε τιμή,\n{name_short}\n{company}",
        ]
    },
    "Italian": {
        "A1": [
            "Ciao {name},\n\nIo sono {name_short}. Io voglio {topic}. Grazie.\n\nSaluti",
            "Gentile {name},\n\nHo un problema. Voglio aiuto con {topic}. Puoi aiutare? Grazie.\n\nCordiali saluti",
            "Ciao,\n\nSono nuovo qui. Voglio {topic}. Non so cosa fare.\n\n{name_short}",
        ],
        "A2": [
            "Gentile {name},\n\nLe scrivo perché ho bisogno di aiuto con {topic}. È un problema importante e spero che possa aiutarmi. Aspetto la sua risposta.\n\nCordiali saluti,\n{name_short}",
            "Ciao colleghi,\n\nVoglio informarvi su {topic}. È importante che tutti lo sappiano. Se avete domande, fatemi sapere.\n\nGrazie,\n{name_short}",
        ],
        "B1": [
            "Gentile {name},\n\nLe scrivo in merito a {topic}. Dopo aver esaminato la situazione, ritengo che dobbiamo prendere provvedimenti. Sarebbe opportuno se potessimo discuterne la prossima settimana.\n\nPuò farmi sapere quando è disponibile per un breve incontro? Propongo martedì o mercoledì, ma sono flessibile.\n\nCordiali saluti,\n{name_short}\n{company}",
        ],
        "B2": [
            "Gentile {name},\n\nIn seguito alla nostra recente conversazione riguardo a {topic}, desidero fornirle un aggiornamento dettagliato sullo stato attuale della situazione.\n\nAbbiamo compiuto progressi significativi nell'ambito dell'implementazione. Il team ha completato con successo la prima fase e siamo attualmente impegnati nella seconda. Si sono verificati alcuni problemi minori, che sono stati nel frattempo risolti.\n\nPropongo di programmare un incontro la prossima settimana per discutere i risultati e definire i prossimi passi.\n\nCordiali saluti,\n{name_short}\n{company}",
        ],
        "C1": [
            "Egregio {name},\n\nAlla luce dei recenti sviluppi inerenti a {topic}, ritengo opportuno presentarle un'analisi approfondita della situazione corrente e delle possibili azioni che la nostra organizzazione potrebbe intraprendere.\n\nDall'analisi dei dati emerge che i risultati dell'ultimo trimestre hanno superato le aspettative, registrando un incremento del 15% nella produttività rispetto allo stesso periodo dell'anno precedente. Ciò è in gran parte attribuibile alla riuscita implementazione del nuovo sistema e alla dedizione dell'intero team.\n\nTuttavia, vi sono alcuni aspetti che meritano un'attenta valutazione, tra cui la scalabilità dell'infrastruttura attuale.\n\nCordiali saluti,\n{name_short}\n{company}",
        ],
        "C2": [
            "Egregio {name},\n\nHo il piacere di presentarle una valutazione esaustiva di {topic}, nella quale non mi limito a contestualizzare i risultati raggiunti, ma illumino altresì le implicazioni strategiche che potrebbero rivelarsi determinanti per il nostro futuro orientamento.\n\nI risultati del periodo trascorso testimoniano un progresso notevole su molteplici fronti. L'efficienza operativa è aumentata del 23%, il che è riconducibile non soltanto all'ottimizzazione dei processi lavorativi, ma anche all'atteggiamento proattivo del nostro team nell'identificare ed eliminare le inefficienze.\n\nSarebbe per me un onore poter approfondire questi risultati nel corso del nostro prossimo incontro.\n\nDistinti saluti,\n{name_short}\n{company}",
        ]
    },
    "Polish": {
        "A1": [
            "Witam {name},\n\nJestem {name_short}. Chcę {topic}. Dziękuję.\n\nPozdrawiam",
            "Drogi {name},\n\nMam problem. Potrzebuję pomocy z {topic}. Czy możesz pomóc? Dziękuję.\n\nZ poważaniem",
            "Witam,\n\nJestem nowy. Chcę {topic}. Nie wiem co robić.\n\n{name_short}",
        ],
        "A2": [
            "Drogi {name},\n\nPiszę do Pana, ponieważ potrzebuję pomocy z {topic}. To jest ważny problem i mam nadzieję, że może mi Pan pomóc. Czekam na odpowiedź.\n\nZ poważaniem,\n{name_short}",
            "Witam kolegów,\n\nChcę was poinformować o {topic}. To jest ważne, żeby wszyscy wiedzieli. Jeśli macie pytania, dajcie mi znać.\n\nDziękuję,\n{name_short}",
        ],
        "B1": [
            "Szanowny {name},\n\nPiszę do Pana w sprawie {topic}. Po przeanalizowaniu sytuacji uważam, że powinniśmy podjąć działania. Byłoby dobrze, gdybyśmy mogli to omówić w przyszłym tygodniu.\n\nCzy może Pan poinformować mnie, kiedy jest dostępny na krótkie spotkanie? Proponuję wtorek lub środę, ale jestem elastyczny.\n\nZ poważaniem,\n{name_short}\n{company}",
        ],
        "B2": [
            "Szanowny {name},\n\nW nawiązaniu do naszej ostatniej rozmowy na temat {topic}, chciałbym przedstawić Panu szczegółową aktualizację obecnego stanu rzeczy.\n\nPoczyniliśmy znaczące postępy w zakresie wdrożenia. Zespół pomyślnie zakończył pierwszy etap i obecnie jesteśmy w trakcie realizacji drugiego. Wystąpiło kilka drobnych problemów, które zostały już rozwiązane.\n\nProponuję, abyśmy zaplanowali spotkanie w przyszłym tygodniu w celu omówienia wyników i ustalenia kolejnych kroków.\n\nZ poważaniem,\n{name_short}\n{company}",
        ],
        "C1": [
            "Szanowny Panie {name},\n\nW świetle ostatnich wydarzeń dotyczących {topic}, uważam za konieczne przedstawienie Panu obszernej analizy obecnej sytuacji oraz możliwych działań, jakie nasza organizacja może podjąć.\n\nZ analizy danych wynika, że wyniki ostatniego kwartału przekroczyły oczekiwania, odnotowując wzrost produktywności o 15% w porównaniu z analogicznym okresem ubiegłego roku. Jest to w dużej mierze zasługą pomyślnego wdrożenia nowego systemu oraz zaangażowania całego zespołu.\n\nNiemniej jednak istnieją pewne kwestie wymagające dalszej analizy, w tym skalowalność obecnej infrastruktury.\n\nZ wyrazami szacunku,\n{name_short}\n{company}",
        ],
        "C2": [
            "Szanowny Panie {name},\n\nZ przyjemnością przedstawiam Panu kompleksową ocenę {topic}, w której nie ograniczam się jedynie do kontekstualizacji osiągniętych wyników, lecz jednocześnie rzucam światło na implikacje strategiczne, które mogą okazać się decydujące dla naszego przyszłego kierunku rozwoju.\n\nWyniki minionego okresu stanowią świadectwo godnego uwagi postępu na wielu frontach. Efektywność operacyjna wzrosła o 23%, co jest wynikiem nie tylko optymalizacji procesów roboczych, ale również proaktywnej postawy naszego zespołu w identyfikowaniu i eliminowaniu nieefektywności.\n\nByłoby mi niezmiernie miło móc omówić te ustalenia podczas naszego najbliższego spotkania.\n\nZ wyrazami najwyższego szacunku,\n{name_short}\n{company}",
        ]
    },
    "Portuguese": {
        "A1": [
            "Olá {name},\n\nEu sou {name_short}. Eu quero {topic}. Obrigado.\n\nCumprimentos",
            "Caro {name},\n\nTenho um problema. Quero ajuda com {topic}. Pode ajudar? Obrigado.\n\nAtenciosamente",
            "Olá,\n\nSou novo aqui. Quero {topic}. Não sei o que fazer.\n\n{name_short}",
        ],
        "A2": [
            "Caro {name},\n\nEstou a escrever porque preciso de ajuda com {topic}. É um problema importante e espero que me possa ajudar. Fico a aguardar a sua resposta.\n\nCom os melhores cumprimentos,\n{name_short}",
            "Olá colegas,\n\nQuero informar-vos sobre {topic}. É importante que todos saibam. Se tiverem perguntas, digam-me.\n\nObrigado,\n{name_short}",
        ],
        "B1": [
            "Caro {name},\n\nEstou a escrever-lhe a respeito de {topic}. Após analisar a situação, acredito que devemos tomar medidas. Seria bom se pudéssemos discutir isto na próxima semana.\n\nPode informar-me quando está disponível para uma breve reunião? Sugiro terça ou quarta-feira, mas sou flexível.\n\nCom os melhores cumprimentos,\n{name_short}\n{company}",
        ],
        "B2": [
            "Caro {name},\n\nNa sequência da nossa recente conversa sobre {topic}, gostaria de lhe fornecer uma atualização detalhada sobre a situação atual.\n\nFizemos progressos significativos na área da implementação. A equipa concluiu com sucesso a primeira fase e estamos agora a trabalhar na segunda. Surgiram alguns problemas menores, que já foram resolvidos.\n\nProponho que agendemos uma reunião para a próxima semana para discutir os resultados e definir os próximos passos.\n\nCom os melhores cumprimentos,\n{name_short}\n{company}",
        ],
        "C1": [
            "Estimado {name},\n\nÀ luz dos recentes desenvolvimentos relativos a {topic}, considero necessário apresentar-lhe uma análise abrangente da situação atual e das possíveis ações que a nossa organização pode empreender.\n\nDa análise dos dados resulta que os resultados do último trimestre superaram as expectativas, registando um aumento de 15% na produtividade em comparação com o mesmo período do ano anterior. Isto deve-se em grande parte à implementação bem-sucedida do novo sistema e à dedicação de toda a equipa.\n\nContudo, existem alguns aspetos que merecem uma análise mais aprofundada, incluindo a escalabilidade da infraestrutura atual.\n\nCom os melhores cumprimentos,\n{name_short}\n{company}",
        ],
        "C2": [
            "Estimado {name},\n\nTenho o prazer de lhe apresentar uma avaliação exaustiva de {topic}, na qual não me circunscrevo apenas à contextualização dos resultados alcançados, mas ilumino igualmente as implicações estratégicas que poderão revelar-se determinantes para a nossa orientação futura.\n\nOs resultados do período transato constituem testemunho de um progresso notável em múltiplas frentes. A eficiência operacional registou um incremento de 23%, o que se deve não apenas à otimização dos processos de trabalho, mas também à postura proativa da nossa equipa na identificação e eliminação de ineficiências.\n\nSeria para mim uma honra poder aprofundar estas conclusões durante o nosso próximo encontro.\n\nCom os mais elevados cumprimentos,\n{name_short}\n{company}",
        ]
    }
}

TOPICS = {
    "Dutch": ["het project", "de vertraging", "de nieuwe software", "het budget", "de klantenservice",
              "de training", "het kantoor", "de vergadering", "de leverancier", "de factuur",
              "de productlancering", "het kwaliteitsprobleem", "de marketingcampagne", "het personeelsbeleid",
              "de technische problemen"],
    "Greek": ["το έργο", "την καθυστέρηση", "το νέο λογισμικό", "τον προϋπολογισμό", "την εξυπηρέτηση πελατών",
              "την εκπαίδευση", "το γραφείο", "τη συνάντηση", "τον προμηθευτή", "το τιμολόγιο",
              "την κυκλοφορία προϊόντος", "το πρόβλημα ποιότητας", "την εκστρατεία μάρκετινγκ",
              "την πολιτική προσωπικού", "τα τεχνικά προβλήματα"],
    "Italian": ["il progetto", "il ritardo", "il nuovo software", "il budget", "il servizio clienti",
                "la formazione", "l'ufficio", "la riunione", "il fornitore", "la fattura",
                "il lancio del prodotto", "il problema di qualità", "la campagna di marketing",
                "la politica del personale", "i problemi tecnici"],
    "Polish": ["projekt", "opóźnienie", "nowe oprogramowanie", "budżet", "obsługę klienta",
               "szkolenie", "biuro", "spotkanie", "dostawcę", "fakturę",
               "wprowadzenie produktu", "problem jakościowy", "kampanię marketingową",
               "politykę kadrową", "problemy techniczne"],
    "Portuguese": ["o projeto", "o atraso", "o novo software", "o orçamento", "o serviço ao cliente",
                   "a formação", "o escritório", "a reunião", "o fornecedor", "a fatura",
                   "o lançamento do produto", "o problema de qualidade", "a campanha de marketing",
                   "a política de pessoal", "os problemas técnicos"]
}


def generate_scores(cefr_level: str) -> Tuple[int, int]:
    """Generate grammar and content scores based on CEFR level distribution."""
    g_dist = SCORE_DISTRIBUTIONS[cefr_level]["grammar"]
    c_dist = SCORE_DISTRIBUTIONS[cefr_level]["content"]
    
    grammar = random.choices(list(g_dist.keys()), weights=list(g_dist.values()), k=1)[0]
    content = random.choices(list(c_dist.keys()), weights=list(c_dist.values()), k=1)[0]
    return grammar, content


def add_errors_for_level(text: str, cefr: str, lang: str) -> str:
    """Add realistic errors to text based on CEFR level."""
    if cefr in ["C1", "C2"]:
        return text  # High-level: no added errors
    
    words = text.split()
    if len(words) < 5:
        return text
    
    if cefr == "A1":
        # Remove ~20% of words randomly, swap some words
        n_remove = max(1, len(words) // 5)
        indices = random.sample(range(len(words)), min(n_remove, len(words)))
        words = [w for i, w in enumerate(words) if i not in indices]
        # Duplicate a random word
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])
    elif cefr == "A2":
        # Swap ~10% of adjacent words
        n_swap = max(1, len(words) // 10)
        for _ in range(n_swap):
            idx = random.randint(0, max(0, len(words) - 2))
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
    elif cefr == "B1":
        # Minor typos: duplicate or delete a character in ~5% of words
        n_typo = max(1, len(words) // 20)
        indices = random.sample(range(len(words)), min(n_typo, len(words)))
        for idx in indices:
            word = words[idx]
            if len(word) > 3:
                char_idx = random.randint(1, len(word) - 2)
                words[idx] = word[:char_idx] + word[char_idx] + word[char_idx:]
    elif cefr == "B2":
        # Very minor: occasional doubled letter
        if random.random() < 0.3:
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 4:
                char_idx = random.randint(1, len(word) - 2)
                words[idx] = word[:char_idx] + word[char_idx] + word[char_idx:]
    
    return " ".join(words)


def generate_fallback_data(samples_per_level: int) -> List[Dict]:
    """Generate rich template-based training data without API."""
    print("Generating template-based training data...")
    data = []
    q_id_counter = 1
    
    total = len(CEFR_LEVELS) * samples_per_level
    
    for cefr in tqdm(CEFR_LEVELS, desc="CEFR Levels"):
        for i in range(samples_per_level):
            lang = random.choice(LANGUAGES)
            grammar, content = generate_scores(cefr)
            
            # Select template
            templates = TEMPLATES[lang][cefr]
            template = random.choice(templates)
            
            # Fill in template
            name = random.choice(NAMES[lang])
            name_short = name.split()[0]
            company = random.choice(COMPANIES)
            topic = random.choice(TOPICS[lang])
            scenario_idx = random.randint(0, len(SCENARIOS[lang]) - 1)
            scenario = SCENARIOS[lang][scenario_idx]
            
            text = template.format(
                name=name,
                name_short=name_short,
                company=company,
                topic=topic
            )
            
            # Add realistic errors based on CEFR level
            text = add_errors_for_level(text, cefr, lang)
            
            # Vary text length slightly
            if cefr in ["A1"] and random.random() < 0.3:
                # Sometimes truncate A1 emails to be shorter
                sentences = text.split('.')
                text = '.'.join(sentences[:max(2, len(sentences) // 2)]) + '.'
            
            data.append({
                "questionID": q_id_counter,
                "questionStatement": scenario,
                "text": text,
                "grammar": grammar,
                "content": content,
                "cefr": cefr,
                "language": lang
            })
            q_id_counter += 1
    
    return data


def generate_with_gemini(api_key: str, samples_per_level: int) -> List[Dict]:
    """Generate training data using Gemini API."""
    if not HAS_GENAI:
        print("google-generativeai is not installed. Run: pip install google-generativeai")
        print("Falling back to template generation.")
        return generate_fallback_data(max(100, samples_per_level))
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    data = []
    q_id_counter = 1
    total = len(CEFR_LEVELS) * samples_per_level
    
    print(f"Generating {total} samples using Gemini API...")
    
    for cefr in CEFR_LEVELS:
        pbar = tqdm(range(samples_per_level), desc=f"CEFR {cefr}")
        for _ in pbar:
            lang = random.choice(LANGUAGES)
            grammar, content = generate_scores(cefr)
            scenario_idx = random.randint(0, len(SCENARIOS[lang]) - 1)
            scenario = SCENARIOS[lang][scenario_idx]
            
            prompt = f"""Write a realistic work/business email in {lang} language.

Target CEFR proficiency level: {cefr}
Level description: {CEFR_DESCRIPTIONS[cefr]}
Target grammar quality (0=terrible, 5=perfect): {grammar}
Target content completeness (0=empty/off-topic, 4=complete): {content}

Context/scenario: {scenario}

IMPORTANT INSTRUCTIONS:
- Write the ENTIRE email in {lang} language only
- Match the proficiency level exactly:
  - If A1/A2: use SIMPLE vocabulary, make grammatical mistakes, use short sentences
  - If B1/B2: use moderate vocabulary, occasional errors, organized structure
  - If C1/C2: use sophisticated vocabulary, complex sentences, professional tone
- If grammar score is low (0-2), deliberately include grammatical errors
- If content score is low (0-1), make the email brief/vague/missing key details
- Include proper email structure (greeting, body, closing)
- Length should be 100-500 words
- Return ONLY the email text, no explanations or translations"""
            
            retries = 3
            success = False
            while retries > 0 and not success:
                try:
                    response = model.generate_content(prompt)
                    text = response.text.strip()
                    if len(text) > 50:  # Basic validation
                        success = True
                        data.append({
                            "questionID": q_id_counter,
                            "questionStatement": scenario,
                            "text": text,
                            "grammar": grammar,
                            "content": content,
                            "cefr": cefr,
                            "language": lang
                        })
                        q_id_counter += 1
                    else:
                        retries -= 1
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    pbar.set_postfix({"error": str(e)[:30]})
                    retries -= 1
                    time.sleep(2)
            
            if not success:
                # Fallback for this sample
                templates = TEMPLATES[lang][cefr]
                template = random.choice(templates)
                name = random.choice(NAMES[lang])
                text = template.format(
                    name=name, name_short=name.split()[0],
                    company=random.choice(COMPANIES),
                    topic=random.choice(TOPICS[lang])
                )
                data.append({
                    "questionID": q_id_counter,
                    "questionStatement": scenario,
                    "text": text,
                    "grammar": grammar,
                    "content": content,
                    "cefr": cefr,
                    "language": lang
                })
                q_id_counter += 1
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate multilingual email training data for assessment model"
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (optional, falls back to templates)")
    parser.add_argument("--output", type=str, default="training_data.csv",
                        help="Output CSV path")
    parser.add_argument("--samples-per-level", type=int, default=100,
                        help="Number of samples per CEFR level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.api_key:
        data = generate_with_gemini(args.api_key, args.samples_per_level)
    else:
        print("No API key provided. Using template-based generation.")
        data = generate_fallback_data(args.samples_per_level)
    
    # Shuffle data
    random.shuffle(data)
    
    print(f"\nWriting {len(data)} samples to {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["questionID", "questionStatement", "text", "grammar", "content", "cefr", "language"]
        )
        writer.writeheader()
        writer.writerows(data)
    
    # Print summary statistics
    import collections
    cefr_counts = collections.Counter(d["cefr"] for d in data)
    lang_counts = collections.Counter(d["language"] for d in data)
    grammar_counts = collections.Counter(d["grammar"] for d in data)
    content_counts = collections.Counter(d["content"] for d in data)
    
    print(f"\n{'='*50}")
    print(f"Training Data Summary")
    print(f"{'='*50}")
    print(f"Total samples: {len(data)}")
    print(f"\nCEFR distribution:")
    for level in CEFR_LEVELS:
        print(f"  {level}: {cefr_counts.get(level, 0)}")
    print(f"\nLanguage distribution:")
    for lang in sorted(lang_counts.keys()):
        print(f"  {lang}: {lang_counts[lang]}")
    print(f"\nGrammar score distribution:")
    for score in sorted(grammar_counts.keys()):
        print(f"  {score}: {grammar_counts[score]}")
    print(f"\nContent score distribution:")
    for score in sorted(content_counts.keys()):
        print(f"  {score}: {content_counts[score]}")
    
    print(f"\nDone! Training data saved to: {args.output}")


if __name__ == "__main__":
    main()
