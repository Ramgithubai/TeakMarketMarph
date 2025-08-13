"""
Web Scraping Modules for Business Contact Research
"""

# Make modules importable
from . import web_scraping_module
from . import streamlit_business_researcher
from . import enhanced_government_researcher

__all__ = [
    'web_scraping_module',
    'streamlit_business_researcher', 
    'enhanced_government_researcher'
]

try:
    from .enhanced_timber_business_researcher import EnhancedTimberBusinessResearcher, research_timber_businesses_from_dataframe
    __all__.append('enhanced_timber_business_researcher')
except ImportError:
    pass  # Optional module

try:
    from .justdial_researcher import JustDialStreamlitResearcher
    __all__.append('justdial_researcher')
except ImportError:
    pass  # Optional module