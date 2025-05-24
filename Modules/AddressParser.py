import re
from typing import Tuple, Optional
import pandas as pd

class AddressParser:
    # Compile patterns once at class level rather than for each call
    PATTERNS = [
        # Simple Israel: "Israel"
        re.compile(r'^(?:Address:\s*)?(?P<country>Israel)$', re.IGNORECASE),
        
        # Detailed Israel patterns (combined into one more efficient pattern)
        re.compile(r'^.+(?P<city>Herzeliya|Herzliya Pituach|Hertzelia Pituah)(?:,\s*[^,]+)?,\s*(?P<country>Israel)$', re.IGNORECASE),
        
        # Norway: "Keilalahdentie 2-4, Finnmark, Norway, Norway"
        re.compile(r'^.+(?P<city>Finnmark),\s*(?P<state>\w+),\s*(?P<country>Norway)(?:,\s*\w+)?$', re.IGNORECASE),
        
        # Korea: "Maetandong 416 Suwon, Gyeonggi-do Samsung Medison Bldg., Suwon, South Korea"
        re.compile(r'.*(?P<city>Suwon),\s*(?P<state>Gyeonggi-do).*?(?P<country>South Korea)$', re.IGNORECASE),
        
        # UK: "Berkshire, West Berkshire RG14 2FN, United Kingdom"
        re.compile(r'^(?:Address:\s*)?(?P<city>Berkshire),\s*(?P<state>West Berkshire)\s*[A-Z0-9\s]+,\s*(?P<country>United Kingdom)$', re.IGNORECASE),
        
        # Germany: "3475 Deer Creek Road, Walldorf, 69190, Germany"
        re.compile(r'^.+(?P<city>Walldorf),\s*\d{5},\s*(?P<country>Germany)$', re.IGNORECASE),
        
        # N/A cases: "#N/A, #N/A, #N/A, #N/A"
        re.compile(r'^(?:Address:\s*)?(#N/A\s*,\s*){3}#N/A$')
    ]

    @classmethod
    def parse_address(cls, address: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not address or pd.isna(address):
            return None, None, None

        address = str(address).strip()
        for pattern in cls.PATTERNS:
            match = pattern.search(address)
            if match:
                groups = match.groupdict()
                return (
                    groups.get('city'),
                    groups.get('state'),
                    groups.get('country')
                )
        return None, None, None

def fill_missing_locations(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ['Acquired', 'Acquiring']:
        addr_col = f"{prefix}_Address_HQ"
        city_col = f"{prefix}_City_HQ"
        state_col = f"{prefix}_State_Region_HQ"
        country_col = f"{prefix}_Country_HQ"

        # Create mask once
        mask = df[addr_col].notna() & (
            df[city_col].isna() | 
            df[state_col].isna() | 
            df[country_col].isna()
        )

        # Process only rows that need updating
        for idx in df.index[mask]:
            city, state, country = AddressParser.parse_address(df.at[idx, addr_col])
            
            # Update only if value is missing and we have a parsed value
            if city and pd.isna(df.at[idx, city_col]):
                df.at[idx, city_col] = city
            if state and pd.isna(df.at[idx, state_col]):
                df.at[idx, state_col] = state
            if country and pd.isna(df.at[idx, country_col]):
                df.at[idx, country_col] = country
        df.drop(columns=[addr_col], inplace=True)