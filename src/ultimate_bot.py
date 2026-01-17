#!/usr/bin/env python3
"""
🚀 ULTIMATE POLYMARKET ARBITRAGE BOT v2

Combines ALL winning strategies from Twitter whales:

1. DUTCH BOOK ARBITRAGE - Buy YES + NO when total < $1
2. LATENCY ARBITRAGE - Binance leads Polymarket by 200-500ms  
3. 99¢ SNIPING - Buy near-certain outcomes for guaranteed profit
4. SPORTS ARBITRAGE - Related markets that drift apart

Based on strategies from: swisstony ($3.68M), easyclap ($648k), 
Account88888 ($645k), 0x8dxd ($550k)
"""

import os
import sys
import time
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("ultimate_bot.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("ultimate")


@dataclass
class PricePoint:
    price: float
    timestamp: datetime


@dataclass 
class Market:
    """Polymarket market"""
    id: str
    condition_id: str
    question: str
    yes_price: float
    no_price: float
    tokens: list
    volume: float = 0
    liquidity: float = 0


@dataclass
class Opportunity:
    """Trading opportunity"""
    market: Market
    strategy: str
    direction: str
    expected_profit_pct: float
    confidence: float
    details: str


@dataclass
class Stats:
    start_time: datetime = field(default_factory=datetime.now)
    scans: int = 0
    dutch_book_found: int = 0
    latency_arb_found: int = 0
    snipe_99c_found: int = 0
    trades_attempted: int = 0
    trades_successful: int = 0
    trades_failed: int = 0
    total_profit: float = 0.0
    total_volume: float = 0.0


# Track recently traded markets to avoid spam
TRADE_COOLDOWN_SECONDS = 300  # 5 minutes between trades on same market


class BinanceFeed:
    """Real-time Binance price feed"""
    
    BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker/solusdt@ticker"
    
    def __init__(self):
        self.prices: Dict[str, float] = {"BTC": 0, "ETH": 0, "SOL": 0}
        self.price_history: Dict[str, deque] = {
            "BTC": deque(maxlen=100),
            "ETH": deque(maxlen=100),
            "SOL": deque(maxlen=100),
        }
        self.running = False
        self._task = None
    
    async def start(self):
        if not WEBSOCKETS_AVAILABLE:
            return
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("📡 Binance feed started")
    
    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def _run(self):
        while self.running:
            try:
                async with websockets.connect(self.BINANCE_WS) as ws:
                    while self.running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)
                        self._process(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(1)
    
    def _process(self, data: dict):
        if "data" not in data:
            return
        ticker = data["data"]
        symbol = ticker.get("s", "")
        price = float(ticker.get("c", 0))
        now = datetime.now()
        
        if "BTCUSDT" in symbol:
            self.prices["BTC"] = price
            self.price_history["BTC"].append(PricePoint(price, now))
        elif "ETHUSDT" in symbol:
            self.prices["ETH"] = price
            self.price_history["ETH"].append(PricePoint(price, now))
        elif "SOLUSDT" in symbol:
            self.prices["SOL"] = price
            self.price_history["SOL"].append(PricePoint(price, now))
    
    def get_price_change(self, asset: str, seconds: float = 15) -> float:
        history = self.price_history.get(asset, deque())
        if len(history) < 2:
            return 0.0
        
        now = datetime.now()
        cutoff = now - timedelta(seconds=seconds)
        
        old_price = None
        for point in history:
            if point.timestamp >= cutoff:
                old_price = point.price
                break
        
        if not old_price:
            old_price = history[0].price if history else 0
        
        current = self.prices.get(asset, old_price)
        if old_price == 0:
            return 0.0
        
        return ((current - old_price) / old_price) * 100


class UltimateBot:
    """Ultimate Polymarket Arbitrage Bot"""
    
    DUTCH_BOOK_THRESHOLD = 0.995  # YES + NO < 99.5¢ for arbitrage
    SNIPE_THRESHOLD = 0.94       # 94¢ or higher for sniping (more opportunities)
    LATENCY_THRESHOLD = 1.0      # 1.0% move for latency arb (more sensitive)
    
    def __init__(self):
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.api_key = os.getenv("POLYMARKET_API_KEY", "")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET", "")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")
        self.proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        
        self.order_size = float(os.getenv("ORDER_SIZE", "5"))
        self.dry_run = os.getenv("DRY_RUN", "True").lower() == "true"
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "10"))
        
        if not all([self.private_key, self.api_key, self.proxy_address]):
            raise ValueError("Missing credentials! Make sure PRIVATE_KEY, POLYMARKET_API_KEY, and POLYMARKET_PROXY_ADDRESS are set.")
        
        if not self.private_key.startswith("0x"):
            self.private_key = "0x" + self.private_key
        
        # signature_type=2 for browser wallet (MetaMask) connected to Polymarket
        # funder=proxy_address because funds are in the Polymarket proxy wallet
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=self.private_key,
            creds=ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            ),
            signature_type=2,  # GNOSIS_SAFE for browser wallet
            funder=self.proxy_address,  # Polymarket proxy wallet address
        )
        
        self.binance = BinanceFeed()
        self.running = False
        self.stats = Stats()
        self.markets: List[Market] = []
        self.daily_loss = 0.0
        self.recently_traded: Dict[str, datetime] = {}  # market_id -> last_trade_time
    
    async def run(self):
        self.running = True
        self._print_startup()
        
        await self.binance.start()
        
        try:
            while self.running:
                if self.daily_loss >= self.max_daily_loss:
                    logger.warning("Daily loss limit reached!")
                    await asyncio.sleep(60)
                    continue
                
                # Fetch fresh market data
                self._fetch_markets()
                
                # Scan for opportunities
                opportunities = self._scan_all()
                
                if opportunities:
                    best = max(opportunities, key=lambda x: x.expected_profit_pct * x.confidence)
                    await self._execute(best)
                
                self.stats.scans += 1
                if self.stats.scans % 30 == 0:
                    self._print_status()
                
                await asyncio.sleep(0.5)  # Scan every 500ms
                
        except asyncio.CancelledError:
            pass
        finally:
            await self.binance.stop()
            self._print_final()
    
    def stop(self):
        self.running = False
    
    def _print_startup(self):
        mode = "🧪 DRY RUN" if self.dry_run else "🔴 LIVE"
        console.print(Panel(
            f"[bold cyan]🚀 ULTIMATE POLYMARKET BOT v2[/bold cyan]\n\n"
            f"Mode: [bold]{mode}[/bold]\n"
            f"Order Size: ${self.order_size:.2f}\n"
            f"Max Daily Loss: ${self.max_daily_loss:.2f}\n\n"
            f"[green]Strategies:[/green]\n"
            f"  ✓ Dutch Book (YES+NO < ${self.DUTCH_BOOK_THRESHOLD})\n"
            f"  ✓ Binance Latency Arb ({self.LATENCY_THRESHOLD}% move)\n"
            f"  ✓ 99¢ Sniping (outcomes > ${self.SNIPE_THRESHOLD})",
            title="Configuration",
            border_style="cyan",
        ))
    
    def _fetch_markets(self):
        """Fetch ALL active markets from Polymarket"""
        try:
            all_markets = []
            next_cursor = "MA=="  # Start cursor
            
            # Fetch markets from CLOB API (paginated)
            for _ in range(10):  # Max 10 pages (~1000 markets)
                try:
                    resp = requests.get(
                        f"https://clob.polymarket.com/markets?next_cursor={next_cursor}",
                        timeout=10,
                    )
                    if resp.status_code != 200:
                        break
                    
                    data = resp.json()
                    markets_page = data if isinstance(data, list) else data.get("data", [])
                    
                    for m in markets_page:
                        # Only active markets
                        if m.get("active") and m.get("accepting_orders"):
                            all_markets.append(m)
                    
                    # Get next cursor
                    next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
                    if not next_cursor or next_cursor == "MA==":
                        break
                        
                except Exception:
                    break
            
            logger.info(f"Found {len(all_markets)} active markets")
            
            self.markets = []
            for m in all_markets:
                try:
                    tokens = m.get("tokens", [])
                    if len(tokens) < 2:
                        continue
                    
                    # Get prices from tokens
                    yes_price = float(tokens[0].get("price", 0.5))
                    no_price = float(tokens[1].get("price", 0.5))
                    
                    # Skip if prices are invalid
                    if yes_price <= 0 or no_price <= 0:
                        continue
                    
                    market = Market(
                        id=m.get("condition_id", ""),
                        condition_id=m.get("condition_id", ""),
                        question=m.get("question", ""),
                        yes_price=yes_price,
                        no_price=no_price,
                        tokens=tokens,
                        volume=float(m.get("volume", 0) or 0),
                        liquidity=float(m.get("liquidity", 0) or 0),
                    )
                    self.markets.append(market)
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
    
    def _is_on_cooldown(self, market_id: str) -> bool:
        """Check if we recently traded this market"""
        if market_id not in self.recently_traded:
            return False
        
        last_trade = self.recently_traded[market_id]
        elapsed = (datetime.now() - last_trade).total_seconds()
        return elapsed < TRADE_COOLDOWN_SECONDS
    
    def _scan_all(self) -> List[Opportunity]:
        """Scan all strategies"""
        opportunities = []
        
        # Clean up old cooldowns
        now = datetime.now()
        self.recently_traded = {
            k: v for k, v in self.recently_traded.items()
            if (now - v).total_seconds() < TRADE_COOLDOWN_SECONDS
        }
        
        for market in self.markets:
            # Skip if we recently traded this market
            if self._is_on_cooldown(market.id):
                continue
            
            # Strategy 1: Dutch Book (PRIORITY - guaranteed profit)
            opp = self._check_dutch_book(market)
            if opp:
                opportunities.append(opp)
                self.stats.dutch_book_found += 1
                continue  # Don't check other strategies for this market
            
            # Strategy 2: 99¢ Snipe
            opp = self._check_99c_snipe(market)
            if opp:
                opportunities.append(opp)
                self.stats.snipe_99c_found += 1
            
            # Strategy 3: Latency Arb (for crypto markets)
            opp = self._check_latency_arb(market)
            if opp:
                opportunities.append(opp)
                self.stats.latency_arb_found += 1
        
        return opportunities
    
    def _check_dutch_book(self, market: Market) -> Optional[Opportunity]:
        """Check for Dutch Book arbitrage (YES + NO < $1)"""
        total = market.yes_price + market.no_price
        
        if total >= self.DUTCH_BOOK_THRESHOLD:
            return None
        
        profit_pct = (1 - total) * 100
        
        return Opportunity(
            market=market,
            strategy="dutch_book",
            direction="BOTH",
            expected_profit_pct=profit_pct,
            confidence=0.99,
            details=f"YES={market.yes_price:.3f} + NO={market.no_price:.3f} = {total:.3f}",
        )
    
    def _check_99c_snipe(self, market: Market) -> Optional[Opportunity]:
        """Check for 99¢ sniping opportunity"""
        # Need valid tokens to trade
        if len(market.tokens) < 2:
            return None
        
        if market.yes_price >= self.SNIPE_THRESHOLD:
            profit_pct = (1 - market.yes_price) * 100
            return Opportunity(
                market=market,
                strategy="99c_snipe",
                direction="YES",
                expected_profit_pct=profit_pct,
                confidence=market.yes_price,  # Higher price = higher confidence
                details=f"YES at ${market.yes_price:.3f} - {profit_pct:.1f}% profit on win",
            )
        
        if market.no_price >= self.SNIPE_THRESHOLD:
            profit_pct = (1 - market.no_price) * 100
            return Opportunity(
                market=market,
                strategy="99c_snipe",
                direction="NO",
                expected_profit_pct=profit_pct,
                confidence=market.no_price,
                details=f"NO at ${market.no_price:.3f} - {profit_pct:.1f}% profit on win",
            )
        
        return None
    
    def _check_latency_arb(self, market: Market) -> Optional[Opportunity]:
        """Check for latency arbitrage on crypto markets"""
        if not WEBSOCKETS_AVAILABLE:
            return None
        
        q = market.question.lower()
        
        # Determine if this is a crypto market
        asset = None
        if "bitcoin" in q or "btc" in q:
            asset = "BTC"
        elif "ethereum" in q or "eth" in q:
            asset = "ETH"
        elif "solana" in q or "sol" in q:
            asset = "SOL"
        
        if not asset:
            return None
        
        # Get Binance price change
        change = self.binance.get_price_change(asset, seconds=15)
        
        if abs(change) < self.LATENCY_THRESHOLD:
            return None
        
        # Check if odds are stale (still near 50/50)
        odds_diff = abs(market.yes_price - market.no_price)
        if odds_diff > 0.2:  # Already repriced
            return None
        
        # Determine direction
        if change > 0:  # Price went UP
            direction = "YES" if "up" in q else "NO"
            confidence = min(0.9, 0.6 + abs(change) * 0.1)
        else:  # Price went DOWN
            direction = "NO" if "up" in q else "YES"
            confidence = min(0.9, 0.6 + abs(change) * 0.1)
        
        target_price = market.yes_price if direction == "YES" else market.no_price
        expected_profit = (1 - target_price) * confidence * 100
        
        return Opportunity(
            market=market,
            strategy="latency_arb",
            direction=direction,
            expected_profit_pct=expected_profit,
            confidence=confidence,
            details=f"Binance {asset} {change:+.2f}%, odds still {market.yes_price:.2f}/{market.no_price:.2f}",
        )
    
    async def _execute(self, opp: Opportunity):
        """Execute a trade"""
        self.stats.trades_attempted += 1
        
        # Mark this market as recently traded (cooldown)
        self.recently_traded[opp.market.id] = datetime.now()
        
        logger.info(f"🎯 {opp.strategy.upper()}: {opp.market.question[:50]}...")
        logger.info(f"   {opp.direction} | Confidence: {opp.confidence:.1%} | Profit: {opp.expected_profit_pct:.2f}%")
        logger.info(f"   {opp.details}")
        
        if self.dry_run:
            logger.info(f"   [DRY RUN] Would trade ${self.order_size:.2f}")
            self.stats.trades_successful += 1
            self.stats.total_profit += self.order_size * opp.expected_profit_pct / 100
            self.stats.total_volume += self.order_size
            return
        
        # LIVE TRADING
        try:
            if opp.direction == "BOTH":
                await self._execute_dutch_book(opp)
            else:
                await self._execute_single(opp)
        except Exception as e:
            logger.error(f"Trade failed: {e}")
            self.stats.trades_failed += 1
            self.daily_loss += 0.1
    
    async def _execute_dutch_book(self, opp: Opportunity):
        """Execute Dutch Book - buy both sides"""
        tokens = opp.market.tokens
        if len(tokens) < 2:
            logger.error("No tokens found!")
            return
        
        total_cost = opp.market.yes_price + opp.market.no_price
        shares = self.order_size / total_cost
        
        yes_token = tokens[0].get("token_id", "")
        no_token = tokens[1].get("token_id", "")
        
        # Buy YES
        yes_order = OrderArgs(
            token_id=yes_token,
            side=BUY,
            size=shares * opp.market.yes_price,
            price=opp.market.yes_price + 0.01,
        )
        signed = self.client.create_order(yes_order)
        result = self.client.post_order(signed, OrderType.FOK)
        
        if not result.get("orderID"):
            logger.error("YES order failed")
            return
        
        # Buy NO
        no_order = OrderArgs(
            token_id=no_token,
            side=BUY,
            size=shares * opp.market.no_price,
            price=opp.market.no_price + 0.01,
        )
        signed = self.client.create_order(no_order)
        result = self.client.post_order(signed, OrderType.FOK)
        
        if not result.get("orderID"):
            logger.error("NO order failed - exposed!")
            return
        
        profit = shares * (1 - total_cost)
        logger.info(f"✅ Dutch Book complete! Profit: ${profit:.4f}")
        
        self.stats.trades_successful += 1
        self.stats.total_profit += profit
        self.stats.total_volume += self.order_size
    
    async def _execute_single(self, opp: Opportunity):
        """Execute single-side trade"""
        tokens = opp.market.tokens
        if len(tokens) < 2:
            logger.error("   ❌ No tokens available for this market")
            return
        
        idx = 0 if opp.direction == "YES" else 1
        token_id = tokens[idx].get("token_id", "")
        
        if not token_id:
            logger.error("   ❌ Token ID not found")
            return
        
        price = opp.market.yes_price if opp.direction == "YES" else opp.market.no_price
        
        # Round price to 2 decimals, size to 2 decimals (Polymarket requirement)
        # Price must be between 0.01 and 0.99
        order_price = min(0.99, round(price + 0.01, 2))
        order_size = round(self.order_size, 2)
        
        logger.info(f"   📤 Placing order: {opp.direction} @ ${order_price:.2f}, size=${order_size:.2f}")
        logger.info(f"   Token: {token_id[:20]}...")
        
        try:
            order = OrderArgs(
                token_id=token_id,
                side=BUY,
                size=order_size,
                price=order_price,
            )
            signed = self.client.create_order(order)
            result = self.client.post_order(signed, OrderType.FOK)
            
            logger.info(f"   Result: {result}")
            
            if result.get("orderID"):
                logger.info(f"   ✅ Order filled! ID: {result.get('orderID')}")
                self.stats.trades_successful += 1
                self.stats.total_volume += self.order_size
            else:
                logger.error(f"   ❌ Order not filled: {result}")
                self.stats.trades_failed += 1
        except Exception as e:
            logger.error(f"   ❌ Order error: {e}")
            self.stats.trades_failed += 1
    
    def _print_status(self):
        runtime = datetime.now() - self.stats.start_time
        
        table = Table(title="📊 Bot Status", show_header=False)
        table.add_column("", style="cyan")
        table.add_column("")
        
        mode = "[red]LIVE[/red]" if not self.dry_run else "[yellow]DRY RUN[/yellow]"
        
        table.add_row("Mode", mode)
        table.add_row("Runtime", str(runtime).split('.')[0])
        table.add_row("Markets", f"{len(self.markets)}")
        table.add_row("On Cooldown", f"{len(self.recently_traded)}")
        table.add_row("Scans", f"{self.stats.scans:,}")
        table.add_row("Dutch Book", f"[green]{self.stats.dutch_book_found}[/green]")
        table.add_row("99¢ Snipes", f"{self.stats.snipe_99c_found}")
        table.add_row("Latency Arb", f"{self.stats.latency_arb_found}")
        table.add_row("Trades OK/Fail", f"[green]{self.stats.trades_successful}[/green]/[red]{self.stats.trades_failed}[/red]")
        table.add_row("Est. Profit", f"[green]${self.stats.total_profit:.4f}[/green]")
        table.add_row("Volume", f"${self.stats.total_volume:.2f}")
        table.add_row("Daily Loss", f"[red]${self.daily_loss:.2f}[/red]")
        
        if WEBSOCKETS_AVAILABLE and self.binance.prices["BTC"] > 0:
            table.add_row("BTC", f"${self.binance.prices['BTC']:,.2f}")
            table.add_row("ETH", f"${self.binance.prices['ETH']:,.2f}")
        
        console.print(table)
    
    def _print_final(self):
        console.print(Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"Scans: {self.stats.scans:,}\n"
            f"Opportunities:\n"
            f"  Dutch Book: {self.stats.dutch_book_found}\n"
            f"  99¢ Snipes: {self.stats.snipe_99c_found}\n"
            f"  Latency Arb: {self.stats.latency_arb_found}\n"
            f"Trades: {self.stats.trades_successful}/{self.stats.trades_attempted}\n"
            f"Est. Profit: ${self.stats.total_profit:.4f}",
            title="📈 Final",
            border_style="green",
        ))


async def main():
    console.print("""
╔═══════════════════════════════════════════════════════════════════╗
║   🚀  ULTIMATE POLYMARKET BOT v2  🚀                              ║
║                                                                   ║
║   Strategies: Dutch Book | Latency Arb | 99¢ Sniping              ║
║   Based on: swisstony, easyclap, Account88888, 0x8dxd             ║
╚═══════════════════════════════════════════════════════════════════╝
    """, style="bold cyan")
    
    bot = UltimateBot()
    signal.signal(signal.SIGINT, lambda s, f: bot.stop())
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
