# metadata/sector.py

# 83-sector classification
SECTOR_83 = {
    1: "Crops",
    2: "Animals",
    3: "Forest goods",
    4: "Fishery goods",
    5: "Agriculture, forestry and fishing related services",
    6: "Coal, crude petroleum and natural gas",
    7: "Metal ores and non-metallic minerals",
    8: "Foods",
    9: "Beverages",
    10: "Tobacco products",
    11: "Textiles and apparels",
    12: "Leather products",
    13: "Wood and wooden products",
    14: "Pulp and paper products",
    15: "Printing and reproduction of recorded media",
    16: "Petroleum and coal products",
    17: "Basic chemical products",
    18: "Synthetic resins and synthetic rubbers",
    19: "Chemical fibers",
    20: "Medicaments",
    21: "Fertilizers and pesticides",
    22: "Other chemical products",
    23: "Plastic products",
    24: "Rubber products",
    25: "Glass products",
    26: "Other non-metallic mineral products",
    27: "Primary iron and steel products",
    28: "Non-ferrous metal ingots and primary Non-ferrous metal products",
    29: "Metal foundries",
    30: "Fabricated metal products, except machinery and furniture",
    31: "Semiconductor and related devices",
    32: "Electronic signal equipment",
    33: "Other electronic components",
    34: "Computer and peripheral equipment",
    35: "Telecommunication, video, and audio equipment",
    36: "Precision instruments",
    37: "Electrical equipment",
    38: "General-purpose machinery and equipment",
    39: "Special-purpose machinery and equipment",
    40: "Motor vehicles",
    41: "Ships",
    42: "Other transport equipment",
    43: "Other manufactured products",
    44: "Manufacturing services and repair services of industrial equipment",
    45: "Electricity supply",
    46: "Gas, steam, hot water supply",
    47: "Water supply",
    48: "Sewage and wastewater treatment services",
    49: "Waste treatment and disposal services",
    50: "Constructions and repairs of buildings",
    51: "Civil engineering",
    52: "Wholesale and retail trade and commodity brokerage services",
    53: "Land transport services",
    54: "Water transport services",
    55: "Air transport services",
    56: "Storage services and supporting services for transportation",
    57: "Postal services and transport services of parcels",
    58: "Food services and accommodation",
    59: "Communications",
    60: "Broadcasting",
    61: "Information services",
    62: "Computer software development and other IT services",
    63: "Newspaper and publishing",
    64: "Video and audio production and distribution",
    65: "Financial services",
    66: "Insurance services",
    67: "Services auxiliary to financial and insurance services",
    68: "Rental or leasing services of residential property",
    69: "Other real estate services",
    70: "Research and development services",
    71: "Business-related professional services",
    72: "Scientific, technical, and other professional services",
    73: "Leasing or rental services concerning equipment, goods and intellectual property rights",
    74: "Business support services",
    75: "Public administration, defense, and social security services",
    76: "Education services",
    77: "Medical and health care services",
    78: "Social care services",
    79: "Cultural- and tour-related services",
    80: "Sports, amusement and recreational services",
    81: "Services of membership organizations",
    82: "Repair and other personal services",
    83: "Others",
}

# 76-sector classification
SECTOR_76 = {
    1: "Crops, animals, and forest goods",
    4: "Agriculture, forestry, fishery goods, and fishing related services",
    6: "Coal, crude petroleum and natural gas",
    7: "Metal ores and non-metallic minerals",
    8: "Foods",
    9: "Beverages",
    10: "Tobacco products",
    11: "Textiles and apparels",
    12: "Leather products",
    13: "Wood and wooden products",
    14: "Pulp and paper products",
    15: "Printing and reproduction of recorded media",
    16: "Petroleum and coal products",
    17: "Basic chemical products",
    18: "Synthetic resins and synthetic rubbers",
    19: "Chemical fibers",
    20: "Medicaments",
    21: "Fertilizers and pesticides",
    22: "Other chemical products",
    23: "Plastic products",
    24: "Rubber products",
    25: "Glass products",
    26: "Other non-metallic mineral products",
    27: "Primary iron and steel products",
    28: "Non-ferrous metal ingots and primary Non-ferrous metal products",
    29: "Metal foundries",
    30: "Fabricated metal products, except machinery and furniture",
    31: "Semiconductor and related devices",
    32: "Electronic signal equipment and other electronic components",
    34: "Computer and peripheral equipment",
    35: "Telecommunication, video, and audio equipment",
    36: "Precision instruments",
    37: "Electrical equipment",
    38: "General-purpose machinery and equipment",
    39: "Special-purpose machinery and equipment",
    40: "Motor vehicles",
    41: "Ships",
    42: "Other transport equipment",
    43: "Other manufactured products",
    44: "Manufacturing services, personal services, and repair services of industrial equipment",
    45: "Electricity supply",
    46: "Gas, steam, hot water supply",
    47: "Water supply",
    48: "Sewage and wastewater treatment services",
    49: "Waste treatment and disposal services",
    50: "Constructions, building repairs, and civil engineering",
    52: "Wholesale and retail trade and commodity brokerage services",
    53: "Land transport services",
    54: "Water transport services",
    55: "Air transport services",
    56: "Storage services and supporting services for transportation",
    57: "Postal services and transport services of parcels",
    58: "Food services and accommodation",
    59: "Communications",
    60: "Broadcasting",
    61: "Information services",
    62: "Computer software development and other IT services",
    63: "Newspaper and publishing",
    64: "Video and audio production and distribution",
    65: "Financial services",
    66: "Insurance services",
    67: "Services auxiliary to financial and insurance services",
    68: "Rental or leasing services of residential property",
    69: "Other real estate services",
    70: "Research and development services",
    71: "Business-related professional services",
    72: "Scientific, technical, and other professional services",
    73: "Leasing or rental services concerning equipment, goods and intellectual property rights",
    74: "Business support services",
    75: "Public administration, defense, and social security services",
    76: "Education services",
    77: "Medical and health care services",
    78: "Social care services",
    79: "Culture, tourism, sports, amusement and recreational services",
    81: "Services of membership organizations",
    83: "Others",
}


# Mapping: 83 -> 76
SECTOR_MAPPING_83_TO_76 = {
    # Agriculture, forestry, livestock (aggregation)
    1: 1,   # Crops, animals, and forest goods → Crops, Forestry Products, and Livestock
    2: 1,   # Animals → Crops, Forestry Products, and Livestock
    3: 1,   # Forest goods → Crops, Forestry Products, and Livestock

    # Agriculture, forestry, fishery goods & services (aggregation)
    4: 4,   # Agriculture, forestry, fishery goods, and fishing related services → Fishery Products and A,F,F Services
    5: 4,   # Agriculture, forestry and fishing related services → Fishery Products and A,F,F Services

    # Mining
    6: 6,   # Coal, crude petroleum and natural gas → Coal, Crude Oil, and Natural Gas
    7: 7,   # Metal ores and non-metallic minerals → Metallic and Non-metallic Minerals

    # Manufacturing
    8: 8,   # Foods → Food Products
    9: 9,   # Beverages → Beverages
    10: 10, # Tobacco products → Tobacco Products
    11: 11, # Textiles and apparels → Textiles and Apparel
    12: 12, # Leather products → Leather Products
    13: 13, # Wood and wooden products → Wood and Wood Products
    14: 14, # Pulp and paper products → Pulp and Paper Products
    15: 15, # Printing and reproduction of recorded media → Printing and Reproduction of Recorded Media
    16: 16, # Petroleum and coal products → Coke and Refined Petroleum Products
    17: 17, # Basic chemical products → Basic Chemicals
    18: 18, # Synthetic resins and synthetic rubbers → Synthetic Resins and Synthetic Rubber
    19: 19, # Chemical fibers → Chemical Fibers
    20: 20, # Medicaments → Pharmaceutical Products
    21: 21, # Fertilizers and pesticides → Fertilizers and Agrochemicals
    22: 22, # Other chemical products → Other Chemical Products
    23: 23, # Plastic products → Plastic Products
    24: 24, # Rubber products → Rubber Products
    25: 25, # Glass products → Glass and Glass Products
    26: 26, # Other non-metallic mineral products → Other Non-metallic Mineral Products
    27: 27, # Primary iron and steel products → Primary Iron and Steel Products
    28: 28, # Non-ferrous metal ingots and primary Non-ferrous metal products → Primary Non-ferrous Metal Products
    29: 29, # Metal foundries → Metal Castings
    30: 30, # Fabricated metal products, except machinery and furniture → Fabricated Metal Products

    # Electronics (aggregation)
    31: 31, # Semiconductor and related devices → Semiconductors
    32: 32, # Electronic signal equipment → Electronic Display Devices and Other Electronic Components
    33: 32, # Other electronic components → Electronic Display Devices and Other Electronic Components

    34: 34, # Computer and peripheral equipment → Computers and Peripheral Equipment
    35: 35, # Telecommunication, video, and audio equipment → Communication, Broadcasting, and Audio-visual Equipment
    36: 36, # Precision instruments → Precision Instruments
    37: 37, # Electrical equipment → Electrical Equipment
    38: 38, # General-purpose machinery and equipment → General-purpose Machinery
    39: 39, # Special-purpose machinery and equipment → Special-purpose Machinery
    40: 40, # Motor vehicles → Motor Vehicles
    41: 41, # Ships → Ships and Boats
    42: 42, # Other transport equipment → Other Transport Equipment
    43: 43, # Other manufactured products → Other Manufactured Products

    # Manufacturing & personal repair services (aggregation)
    44: 44, # Manufacturing services and repair services of industrial equipment → Manufacturing Services and Industrial Machinery Repair, Personal Services
    82: 44, # Repair and other personal services → Manufacturing Services and Industrial Machinery Repair, Personal Services

    # Energy & utilities
    45: 45, # Electricity supply → Electric Power and Renewable Energy
    46: 46, # Gas, steam, hot water supply → Gas, Steam, and Hot Water Supply
    47: 47, # Water supply → Water Supply
    48: 48, # Sewage and wastewater treatment services → Sewage Treatment
    49: 49, # Waste treatment and disposal services → Waste Treatment and Resource Recycling Services

    # Construction (aggregation)
    50: 50, # Constructions and repairs of buildings → Building Construction, Building Repair, and Civil Engineering
    51: 50, # Civil engineering → Building Construction, Building Repair, and Civil Engineering

    # Trade & transport
    52: 52, # Wholesale and retail trade and commodity brokerage services → same
    53: 53, # Land transport services → Land Transport Services
    54: 54, # Water transport services → Water Transport Services
    55: 55, # Air transport services → Air Transport Services
    56: 56, # Storage services and supporting services for transportation → Warehousing and Transport Support Services
    57: 57, # Postal services and transport services of parcels → Postal and Courier Services

    # Services
    58: 58, # Food services and accommodation → Food and Beverage Services and Accommodation
    59: 59, # Communications → Telecommunication Services
    60: 60, # Broadcasting → Broadcasting Services
    61: 61, # Information services → Information Services
    62: 62, # Computer software development and other IT services → Software Development, Supply, and Other IT Services
    63: 63, # Newspaper and publishing → Newspaper and Publishing Services
    64: 64, # Video and audio production and distribution → Motion Picture and Audio Production and Distribution
    65: 65, # Financial services → Financial Services
    66: 66, # Insurance services → Insurance Services
    67: 67, # Services auxiliary to financial and insurance services → Financial and Insurance Auxiliary Services
    68: 68, # Rental or leasing services of residential property → Residential Services
    69: 69, # Other real estate services → Other Real Estate Services
    70: 70, # Research and development services → Research and Development
    71: 71, # Business-related professional services → Business-related Professional Services
    72: 72, # Scientific, technical, and other professional services → Scientific, Technical, and Other Professional Services
    73: 73, # Leasing or rental services concerning equipment, goods and intellectual property rights → Rental of Equipment, Goods, and Intellectual Property
    74: 74, # Business support services → Business Support Services
    75: 75, # Public administration, defense, and social security services → Public Administration, Defense, and Social Security
    76: 76, # Education services → Education Services
    77: 77, # Medical and health care services → Health and Medical Services
    78: 78, # Social care services → Social Welfare Services

    # Cultural and sports services (aggregation)
    79: 79, # Cultural- and tour-related services → Cultural, Tourism, and Sports-related Services
    80: 79, # Sports, amusement and recreational services → Cultural, Tourism, and Sports-related Services

    # Social organizations
    81: 81, # Services of membership organizations → Social Organizations

    # Other
    83: 83, # Others → Other
}