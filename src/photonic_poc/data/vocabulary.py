"""Generic vocabulary for the synthetic generator.

Brand stems are assembled from syllables, so they do not name real companies.
Cities are public geography; postcodes and streets built from them are synthetic.
"""

SYLLABLES: tuple[str, ...] = (
    "vel", "to", "kran", "dor", "mar", "tex", "lum", "nor", "zen", "bri", "tal", "ven",
    "kor", "sil", "ram", "tro", "fen", "gal", "hol", "jun", "pro", "qua", "dra", "mi",
    "lo", "ka", "ne", "su", "ri", "pa", "wen", "fli", "gor", "bax", "cal", "dex", "elm",
    "far", "gri", "han", "ilo", "jor", "kel", "lan", "mon", "nis", "orn", "pel", "rox",
    "sam", "tur", "ulm", "vik", "wal", "xan", "yor", "zol", "bel", "cor", "dan", "eri",
    "fol", "gan", "hal", "ist", "kum", "len", "mur", "nal", "ost", "pun", "ran", "sor",
)  # fmt: skip

INDUSTRY_WORDS: tuple[str, ...] = (
    "Logistik", "Bau", "Software", "Consulting", "Elektro", "Immobilien", "Medizintechnik",
    "Transport", "Metallbau", "Catering", "Reinigung", "Druck", "Maschinenbau", "Autohaus",
    "Energie", "Solar", "Sanitär", "Gartenbau", "Handel", "Pharma", "Marketing", "Design",
    "Personal", "Textil", "Spedition", "Engineering", "Labor", "Versicherungsmakler",
    "Fahrzeugbau", "Dienstleistungen", "Technik", "Haustechnik", "Verpackung", "Recycling",
)  # fmt: skip

# Second words that mark a sister company rather than a different line of business.
SISTER_WORDS: tuple[str, ...] = (
    "Holding", "Verwaltung", "Beteiligungen", "Service", "Vertrieb", "Systems", "Group",
    "Handelsgesellschaft", "Immobilien",
)  # fmt: skip

LEGAL_FORMS: dict[str, tuple[tuple[str, int], ...]] = {
    "DE": (
        ("GmbH", 50), ("GmbH & Co. KG", 15), ("UG (haftungsbeschränkt)", 8), ("AG", 5),
        ("KG", 5), ("OHG", 2), ("e.K.", 6), ("GbR", 4),
    ),
    "AT": (("GmbH", 60), ("AG", 8), ("KG", 8), ("GmbH & Co KG", 12), ("e.U.", 6), ("OG", 6)),
    "CH": (("AG", 55), ("GmbH", 30), ("Sàrl", 10), ("Genossenschaft", 5)),
}  # fmt: skip

# Written variants of a legal form; "" means the form was dropped.
LEGAL_FORM_VARIANTS: dict[str, tuple[str, ...]] = {
    "GmbH & Co. KG": ("GmbH & Co KG", "GmbH u. Co. KG", "GmbH und Co. KG", "GmbH & Co.KG", ""),
    "GmbH & Co KG": ("GmbH & Co. KG", "GmbH u. Co. KG", "GmbH und Co KG", ""),
    "UG (haftungsbeschränkt)": ("UG", "UG (hb)", "UG haftungsbeschränkt", ""),
    "GmbH": ("G.m.b.H.", "GmbH.", "Gmbh", ""),
    "AG": ("A.G.", "Ag", ""),
    "KG": ("K.G.", ""),
    "OHG": ("o.H.G.", ""),
    "e.K.": ("eK", "e. K.", ""),
    "e.U.": ("eU", ""),
    "GbR": ("G.b.R.", ""),
    "OG": ("O.G.", ""),
    "Sàrl": ("SARL", "Sarl", ""),
    "Genossenschaft": ("Gen.", ""),
}  # fmt: skip

STREET_SUFFIXES: tuple[str, ...] = ("straße", "weg", "platz", "allee", "gasse", "ring")
STREET_PREFIXES: tuple[str, ...] = (
    "Haupt", "Bahnhof", "Industrie", "Gewerbe", "Schul", "Markt", "Garten", "Berg", "Linden",
    "Eichen", "Kirch", "Feld", "Wald", "Park", "Mühlen", "Hafen",
)  # fmt: skip

# (city, country, two-digit postcode prefix, area code without leading zero)
CITIES: tuple[tuple[str, str, str, str], ...] = (
    ("Berlin", "DE", "10", "30"), ("Hamburg", "DE", "20", "40"), ("München", "DE", "80", "89"),
    ("Köln", "DE", "50", "221"), ("Frankfurt am Main", "DE", "60", "69"),
    ("Stuttgart", "DE", "70", "711"), ("Düsseldorf", "DE", "40", "211"),
    ("Leipzig", "DE", "04", "341"), ("Dresden", "DE", "01", "351"),
    ("Hannover", "DE", "30", "511"), ("Nürnberg", "DE", "90", "911"),
    ("Bremen", "DE", "28", "421"), ("Essen", "DE", "45", "201"), ("Dortmund", "DE", "44", "231"),
    ("Mannheim", "DE", "68", "621"), ("Karlsruhe", "DE", "76", "721"),
    ("Augsburg", "DE", "86", "821"), ("Wiesbaden", "DE", "65", "611"),
    ("Münster", "DE", "48", "251"), ("Kiel", "DE", "24", "431"), ("Rostock", "DE", "18", "381"),
    ("Erfurt", "DE", "99", "361"), ("Mainz", "DE", "55", "6131"), ("Freiburg", "DE", "79", "761"),
    ("Wien", "AT", "10", "1"), ("Graz", "AT", "80", "316"), ("Linz", "AT", "40", "732"),
    ("Salzburg", "AT", "50", "662"), ("Innsbruck", "AT", "60", "512"),
    ("Zürich", "CH", "80", "44"), ("Bern", "CH", "30", "31"), ("Basel", "CH", "40", "61"),
    ("Genève", "CH", "12", "22"), ("Luzern", "CH", "60", "41"), ("St. Gallen", "CH", "90", "71"),
)  # fmt: skip

COUNTRY_WEIGHTS: tuple[tuple[str, int], ...] = (("DE", 80), ("AT", 10), ("CH", 10))
CALLING_CODES: dict[str, str] = {"DE": "49", "AT": "43", "CH": "41"}
TLDS: dict[str, str] = {"DE": "de", "AT": "at", "CH": "ch"}
