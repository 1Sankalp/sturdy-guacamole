# 🚀 Polymarket Arbitrage Bot

An automated trading bot for [Polymarket](https://polymarket.com) that scans for arbitrage opportunities.

## Strategies

| Strategy | Description | Risk |
|----------|-------------|------|
| **Dutch Book** | Buy YES + NO when total < $1 for guaranteed profit | Low |
| **99¢ Sniping** | Buy near-certain outcomes (96%+) before resolution | Medium |
| **Latency Arb** | Exploit Binance-Polymarket price lag on crypto markets | Medium |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/polystuff.git
cd polystuff
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Get from MetaMask: Settings → Security → Export Private Key
PRIVATE_KEY=your_64_character_key_without_0x

# Your wallet address
WALLET_ADDRESS=0xYourAddress

# Generate these with: python generate_api_keys.py
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=

# Start with DRY_RUN=True to test safely
DRY_RUN=True
ORDER_SIZE=2
MAX_DAILY_LOSS=10
```

### 3. Generate API Keys

```bash
python generate_api_keys.py
```

Copy the output into your `.env` file.

### 4. Run

```bash
# Test mode (no real trades)
python -m src.ultimate_bot

# Live mode (edit .env: DRY_RUN=False)
python -m src.ultimate_bot
```

## Deployment (24/7 VPS)

For low-latency 24/7 operation, deploy to a VPS (DigitalOcean, AWS, etc):

```bash
# On your VPS
git clone YOUR_REPO_URL
cd polystuff
chmod +x deploy.sh
./deploy.sh

# Copy your .env file (never commit it!)
nano .env

# Run with tmux (keeps running after disconnect)
tmux new -s polybot
source venv/bin/activate
python -m src.ultimate_bot
# Press Ctrl+B, then D to detach
```

## ⚠️ Important

- **Start with DRY_RUN=True** to test without risking money
- **Deposit funds to Polymarket** (not just your wallet) before live trading
- **Never commit your .env file** - it contains your private keys
- **Maximum loss** = Whatever you deposit to Polymarket

## File Structure

```
polystuff/
├── src/
│   ├── __init__.py
│   └── ultimate_bot.py    # Main bot
├── .env.example           # Config template
├── .gitignore
├── deploy.sh              # VPS deployment script
├── generate_api_keys.py   # API key generator
├── requirements.txt
└── README.md
```

## License

MIT
