# 🚀 TeakMarketMarph - AI-Powered Business Data Analysis Platform

An intelligent AI-powered data analysis tool for market research with smart data detection, advanced filtering, and business intelligence capabilities. Perfect for analyzing trade data, identifying business patterns, and conducting market research.

🌟 **Live Demo:** [teakmarketmarph.streamlit.app](https://teakmarketmarph.streamlit.app)

## ✨ Current Features (Fully Working)

### 📊 Smart Data Analysis
- **Auto-detects identifier columns** (HS codes, product codes, business IDs)
- **Intelligent data type detection** preserves data integrity for trade analysis
- **Advanced filtering system** with primary/secondary filters and text search
- **Business intelligence metrics** with real-time statistics

### 🤖 AI-Powered Insights
- **Chat with your data** using natural language queries
- **Groq AI integration** for fast, cost-effective analysis
- **Data pattern recognition** and automated insights
- **Custom visualization recommendations**

### 📈 Interactive Visualizations
- **Quick visualization builder** with auto-chart detection
- **Category distribution analysis** for business data
- **Numeric trend analysis** with histogram distributions
- **Export-ready charts** for presentations

### 📁 Smart File Handling
- **CSV and Excel support** with sheet selection
- **Automatic encoding detection** for international data
- **Large file processing** optimized for business datasets
- **Download filtered results** with intelligent naming

## 🚀 Quick Start

### Option 1: Use Online (Recommended)
Visit [teakmarketmarph.streamlit.app](https://teakmarketmarph.streamlit.app) - no setup required!

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/Ramgithubai/TeakMarketMarph.git
cd TeakMarketMarph

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run ai_csv_analyzer.py
```

### Option 3: Deploy Your Own
[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

1. Fork this repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with `ai_csv_analyzer.py` as main file

## 🔐 API Configuration (Optional)

The app works without API keys but provides enhanced features when configured:

```bash
# Copy environment template
cp .env.example .env

# Add your API keys (optional)
GROQ_API_KEY=gsk_your_groq_key_here
```

**Get API keys:**
- [Groq Console](https://console.groq.com/keys) - Fast, cost-effective AI analysis

## 📊 Perfect For

### Trade Data Analysis
- **Import/Export data** with HS code detection
- **Customs data analysis** with automated categorization
- **Supplier/buyer identification** and contact research
- **Trade pattern analysis** with filtering and insights

### Business Intelligence
- **Customer data analysis** with smart segmentation
- **Product catalog analysis** with code detection
- **Market research data** with AI-powered insights
- **Sales data analysis** with trend identification

### Data Exploration
- **Large dataset exploration** with intelligent filtering
- **Pattern discovery** using AI analysis
- **Data quality assessment** with completeness metrics
- **Quick visualization** for presentations

## 🛠️ How It Works

1. **📁 Upload Your Data**
   - Drag & drop CSV or Excel files
   - Automatic data type detection
   - Identifier column recognition

2. **🔍 Explore with Filters**
   - Primary/secondary categorical filters
   - Text search across all columns
   - Real-time filtering with statistics

3. **🤖 AI Analysis**
   - Ask questions in natural language
   - Get insights about patterns and trends
   - Automated data quality assessment

4. **📈 Visualize Results**
   - Auto-generated charts and graphs
   - Interactive visualizations
   - Export-ready presentations

5. **📥 Export Enhanced Data**
   - Download filtered results
   - Enhanced datasets with analysis
   - Professional CSV formatting

## 🏗️ Coming Soon: Business Research Module

We're developing advanced business contact research features:

- 🌐 **Multi-source contact extraction**
- 🏛️ **Government database integration**
- 📞 **JustDial phone number extraction**
- 📧 **Email and website discovery**
- 🔗 **Enhanced dataset exports**

*The core data analysis platform is fully functional and ready for production use.*

## 📁 Project Structure

```
TeakMarketMarph/
├── ai_csv_analyzer.py          # Main application (✅ Working)
├── data_explorer.py            # Advanced data exploration (✅ Working)
├── requirements.txt            # Dependencies (✅ Optimized)
├── .env.example               # Environment template
├── .streamlit/                # Streamlit configuration
└── modules/                   # Business research (🚧 In Development)
```

## 🔧 Technical Details

- **Framework:** Streamlit for web interface
- **Data Processing:** Pandas with intelligent type detection
- **Visualizations:** Plotly for interactive charts
- **AI Integration:** Groq API for natural language analysis
- **Deployment:** Optimized for Streamlit Cloud

## 🤝 Contributing

Contributions welcome! The platform is designed for extensibility:

1. **Core data analysis** - Fully stable
2. **Business research modules** - Under active development
3. **API integrations** - Modular design for easy additions

## 📝 License

MIT License - See LICENSE file for details.

## 💡 Use Cases

**Import/Export Analysis:**
- Analyze customs data with automatic HS code detection
- Identify top trading partners and products
- Track trade patterns over time

**Market Research:**
- Segment customers by behavior patterns
- Analyze product performance across regions
- Discover market opportunities with AI insights

**Business Intelligence:**
- Process large datasets with smart filtering
- Generate insights for executive reporting
- Export analysis-ready data for presentations

---

*Built with ❤️ for data analysts, market researchers, and business intelligence professionals*

**🔗 Links:**
- **Live App:** [teakmarketmarph.streamlit.app](https://teakmarketmarph.streamlit.app)
- **Repository:** [github.com/Ramgithubai/TeakMarketMarph](https://github.com/Ramgithubai/TeakMarketMarph)
- **Issues:** [Report bugs or request features](https://github.com/Ramgithubai/TeakMarketMarph/issues)
