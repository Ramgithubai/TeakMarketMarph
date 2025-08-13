# 🚀 TeakMarketMarph - Advanced Business Contact Research Platform

An intelligent AI-powered web scraping tool for market research with multi-source data extraction, enhanced government research, and JustDial integration. 

Perfect for finding business contacts, analyzing trade data, and conducting comprehensive market research with advanced filtering capabilities.

## ✨ Key Features

- **Smart Data Loading**: Auto-detects identifier columns (HS codes, product codes)
- **AI Chat Interface**: Ask questions about your data in natural language
- **Multiple AI Providers**: Claude, Groq, OpenAI, and local analysis
- **Advanced Filtering**: Business intelligence focused data exploration
- **Automated Research**: Multi-source business contact finding
- **Enhanced Government Research**: Government sources integration
- **JustDial Integration**: Phone number extraction from JustDial
- **Export Capabilities**: Enhanced datasets with research results

## 🛠️ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ramgithubai/TeakMarketMarph.git
   cd TeakMarketMarph
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (Optional - app works without them)
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the application**
   ```bash
   streamlit run ai_csv_analyzer.py
   ```

## 🔐 API Configuration

The app works without API keys but provides enhanced features when configured:

- `GROQ_API_KEY` - For Groq AI chat (Recommended)
- `OPENAI_API_KEY` - For OpenAI GPT chat  
- `TAVILY_API_KEY` - For web scraping research

Get API keys from:
- [Groq Console](https://console.groq.com/keys) (Fast, cost-effective)
- [OpenAI API](https://platform.openai.com/api-keys)
- [Tavily API](https://tavily.com)

## 📊 Usage

1. **Upload CSV/Excel files** - Intelligent data type detection
2. **Explore data** - Advanced filtering and business intelligence
3. **Chat with your data** - Natural language AI analysis
4. **Research business contacts** - Multi-source contact extraction
5. **Export enhanced datasets** - Original data + research results

## 🌟 Special Features

### Enhanced Timber Research
Smart filtering and research specifically designed for teak wood and timber businesses with:
- Product description filtering (TEAK, WOOD, BOARD, TIMBER, LUMBER)
- City-based filtering
- Specialized contact extraction

### Multi-Source Research
- **Standard Research**: Web-based contact extraction
- **Enhanced Research**: Government sources + web research
- **JustDial Research**: Phone number extraction with WhatsApp integration

### Intelligent Data Detection
- Auto-detects HS codes, product codes, and other identifiers
- Preserves data integrity for trade and business analysis
- Smart visualization recommendations

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with `ai_csv_analyzer.py` as main file
4. Add API keys in Streamlit Cloud secrets

### Local Development
```bash
streamlit run ai_csv_analyzer.py
```

## 📁 Project Structure

```
TeakMarketMarph/
├── ai_csv_analyzer.py          # Main application
├── data_explorer.py            # Advanced data exploration
├── modules/
│   ├── web_scraping_module.py  # Core web scraping (FIXED)
│   ├── enhanced_government_researcher.py
│   ├── justdial_researcher.py
│   └── streamlit_business_researcher.py
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables template
└── .streamlit/                # Streamlit configuration
```

## 🔧 Recent Fixes

- ✅ **Fixed `get_selected_city_from_df` error** - Added missing function for filename prefixing
- ✅ **Enhanced filename generation** - Uses primary filter values as prefixes
- ✅ **Improved error handling** - Better debugging and user guidance
- ✅ **Optimized data processing** - Smarter duplicate handling

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.

---

*Built with ❤️ for business intelligence and market research professionals*
