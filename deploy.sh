#!/bin/bash
# =============================================================
# POLYMARKET BOT - VPS DEPLOYMENT SCRIPT
# =============================================================
#
# Run this on your VPS after cloning the repo:
#   chmod +x deploy.sh && ./deploy.sh
#
# =============================================================

set -e

echo "🚀 Polymarket Bot - VPS Deployment"
echo "=================================="

# Check if Python 3.11+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found!"
    echo "   1. Copy .env.example to .env"
    echo "   2. Add your credentials"
    echo ""
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the bot:"
echo "  source venv/bin/activate"
echo "  python -m src.ultimate_bot"
echo ""
echo "To run in background with tmux:"
echo "  tmux new -s polybot"
echo "  source venv/bin/activate"
echo "  python -m src.ultimate_bot"
echo "  # Press Ctrl+B then D to detach"
echo ""
echo "To run as a systemd service (24/7):"
echo "  sudo cp polymarket-bot.service /etc/systemd/system/"
echo "  sudo systemctl enable polymarket-bot"
echo "  sudo systemctl start polymarket-bot"
