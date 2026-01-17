#!/usr/bin/env python3
"""
🚀 ULTIMATE POLYMARKET BOT v3 - AGGRESSIVE MODE 🚀

Strategies:
1. DUTCH BOOK - Buy YES + NO when total < $1 (guaranteed profit)
2. HIGH-CONFIDENCE SNIPE - Buy outcomes at 90%+ (near-certain)
3. VALUE BETTING - Buy underpriced outcomes
4. MOMENTUM - Follow price movements
5. LATENCY ARB - Beat slow market makers

Based on whales: swisstony ($3.68M), easyclap ($648k), 0x8dxd ($550k)
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
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
from py_clob_client.order_builder.constants import BUY

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
    id: str
    condition_id: str
    question: str
    yes_price: float
    no_price: float
    tokens: list
    volume: float = 0
    liquidity: float = 0
    end_date: Optional[datetime] = None


@dataclass
class Opportunity:
    market: Market
    strategy: str
    direction: str
    expected_profit_pct: float
    confidence: float
    details: str
    suggested_size: float = 1.0


@dataclass
class Stats:
    start_time: datetime = field(default_factory=datetime.now)
    scans: int = 0
    opportunities_found: int = 0
    trades_attempted: int = 0
    trades_successful: int = 0
    trades_failed: int = 0
    total_profit: float = 0.0
    total_volume: float = 0.0
    balance: float = 0.0


# Aggressive settings
TRADE_COOLDOWN_SECONDS = 60  # Only 1 minute cooldown
SCAN_INTERVAL = 0.3  # Scan every 300ms
MAX_DAYS_TO_RESOLUTION = 7  # Only quick markets


class BinanceFeed:
    """Real-time Binance price feed for crypto latency arb"""
    
    BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker/solusdt@ticker/xrpusdt@ticker"
    
    def __init__(self):
        self.prices: Dict[str, float] = {"BTC": 0, "ETH": 0, "SOL": 0, "XRP": 0}
        self.price_history: Dict[str, deque] = {k: deque(maxlen=100) for k in self.prices}
        self.running = False
        self._task = None
    
    async def start(self):
        if not WEBSOCKETS_AVAILABLE:
            return
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("📡 Binance WebSocket connected")
    
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
            except Exception:
                await asyncio.sleep(1)
    
    def _process(self, data: dict):
        if "data" not in data:
            return
        ticker = data["data"]
        symbol = ticker.get("s", "")
        price = float(ticker.get("c", 0))
        now = datetime.now()
        
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            if f"{asset}USDT" in symbol:
                self.prices[asset] = price
                self.price_history[asset].append(PricePoint(price, now))
    
    def get_momentum(self, asset: str, seconds: float = 30) -> float:
        """Get price change % over last N seconds"""
        history = self.price_history.get(asset, deque())
        if len(history) < 2:
            return 0.0
        
        cutoff = datetime.now() - timedelta(seconds=seconds)
        old_prices = [p.price for p in history if p.timestamp <= cutoff]
        
        if not old_prices:
            return 0.0
        
        old_price = old_prices[-1] if old_prices else history[0].price
        current = self.prices.get(asset, old_price)
        
        if old_price == 0:
            return 0.0
        
        return ((current - old_price) / old_price) * 100


class UltimateBot:
    """AGGRESSIVE Polymarket Trading Bot"""
    
    # Strategy thresholds - AGGRESSIVE
    DUTCH_BOOK_THRESHOLD = 0.995   # YES + NO < 99.5¢
    SNIPE_THRESHOLD = 0.90         # 90%+ confidence (was 94%)
    VALUE_THRESHOLD = 0.15         # 15%+ edge on value bets
    MOMENTUM_THRESHOLD = 0.8       # 0.8% price move
    
    # Risk settings
    RISK_PER_TRADE = 0.10          # Risk 10% of balance per trade
    MIN_ORDER_SIZE = 1.0           # Minimum $1 per trade
    MAX_ORDER_SIZE = 20.0          # Maximum $20 per trade
    
    def __init__(self):
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.api_key = os.getenv("POLYMARKET_API_KEY", "")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET", "")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")
        self.proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        
        self.dry_run = os.getenv("DRY_RUN", "True").lower() == "true"
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "15"))
        
        if not all([self.private_key, self.api_key, self.proxy_address]):
            raise ValueError("Missing credentials!")
        
        if not self.private_key.startswith("0x"):
            self.private_key = "0x" + self.private_key
        
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=self.private_key,
            creds=ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            ),
            signature_type=2,
            funder=self.proxy_address,
        )
        
        self.binance = BinanceFeed()
        self.running = False
        self.stats = Stats()
        self.markets: List[Market] = []
        self.daily_loss = 0.0
        self.recently_traded: Dict[str, datetime] = {}
        self.price_history: Dict[str, deque] = {}  # Track market price changes
    
    def _get_balance(self) -> float:
        """Get current USDC balance from Polymarket"""
        try:
            # Try to get balance via API
            # For now, we'll estimate based on what we know
            # The real balance check would need the Polymarket balance endpoint
            return max(15.0, self.stats.balance)  # Assume at least $15
        except Exception:
            return 15.0
    
    def _calculate_order_size(self, confidence: float) -> float:
        """Dynamic order sizing based on balance and confidence"""
        balance = self._get_balance()
        
        # Base size is RISK_PER_TRADE of balance
        base_size = balance * self.RISK_PER_TRADE
        
        # Scale by confidence (higher confidence = larger bet)
        size = base_size * confidence
        
        # Clamp to min/max
        size = max(self.MIN_ORDER_SIZE, min(self.MAX_ORDER_SIZE, size))
        
        return round(size, 2)
    
    async def run(self):
        self.running = True
        self._print_startup()
        
        await self.binance.start()
        
        try:
            while self.running:
                if self.daily_loss >= self.max_daily_loss:
                    logger.warning("⚠️ Daily loss limit reached! Pausing...")
                    await asyncio.sleep(300)  # 5 min pause
                    continue
                
                # Fetch markets
                self._fetch_markets()
                
                # Scan ALL strategies
                opportunities = self._scan_all_strategies()
                
                # Execute ALL good opportunities (not just the best)
                for opp in opportunities[:5]:  # Top 5 opportunities
                    await self._execute(opp)
                    await asyncio.sleep(0.1)  # Small delay between orders
                
                self.stats.scans += 1
                if self.stats.scans % 50 == 0:
                    self._print_status()
                
                await asyncio.sleep(SCAN_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        finally:
            await self.binance.stop()
            self._print_final()
    
    def stop(self):
        self.running = False
    
    def _print_startup(self):
        mode = "🧪 DRY RUN" if self.dry_run else "🔴 LIVE TRADING"
        console.print(Panel(
            f"[bold red]🚀 ULTIMATE BOT v3 - AGGRESSIVE MODE 🚀[/bold red]\n\n"
            f"Mode: [bold]{mode}[/bold]\n"
            f"Risk per trade: {self.RISK_PER_TRADE*100:.0f}% of balance\n"
            f"Max daily loss: ${self.max_daily_loss:.2f}\n\n"
            f"[green]Strategies:[/green]\n"
            f"  ✓ Dutch Book (YES+NO < {self.DUTCH_BOOK_THRESHOLD})\n"
            f"  ✓ Snipe ({self.SNIPE_THRESHOLD*100:.0f}%+ outcomes)\n"
            f"  ✓ Value Betting ({self.VALUE_THRESHOLD*100:.0f}%+ edge)\n"
            f"  ✓ Momentum Trading\n"
            f"  ✓ Latency Arbitrage\n\n"
            f"[yellow]Scanning every {SCAN_INTERVAL*1000:.0f}ms[/yellow]",
            title="Configuration",
            border_style="red",
        ))
    
    def _fetch_markets(self):
        """Fetch ALL quick-resolving markets"""
        try:
            all_markets = []
            next_cursor = "MA=="
            now = datetime.now(timezone.utc)
            
            for _ in range(15):  # More pages
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
                        if not (m.get("active") and m.get("accepting_orders")):
                            continue
                        
                        # Check end date
                        end_date = None
                        end_date_str = m.get("end_date_iso") or m.get("game_start_time")
                        if end_date_str:
                            try:
                                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                                days_until = (end_date - now).days
                                if days_until > MAX_DAYS_TO_RESOLUTION or days_until < 0:
                                    continue
                            except Exception:
                                q = m.get("question", "").lower()
                                if any(x in q for x in ["2027", "2028", "2029", "2030"]):
                                    continue
                        
                        all_markets.append((m, end_date))
                    
                    next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
                    if not next_cursor:
                        break
                        
                except Exception:
                    break
            
            self.markets = []
            for m, end_date in all_markets:
                try:
                    tokens = m.get("tokens", [])
                    if len(tokens) < 2:
                        continue
                    
                    yes_price = float(tokens[0].get("price", 0.5))
                    no_price = float(tokens[1].get("price", 0.5))
                    
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
                        end_date=end_date,
                    )
                    self.markets.append(market)
                    
                    # Track price history for momentum
                    if market.id not in self.price_history:
                        self.price_history[market.id] = deque(maxlen=50)
                    self.price_history[market.id].append(
                        PricePoint(yes_price, datetime.now())
                    )
                    
                except Exception:
                    continue
            
            logger.info(f"Found {len(self.markets)} markets (≤{MAX_DAYS_TO_RESOLUTION} days)")
                    
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
    
    def _is_on_cooldown(self, market_id: str) -> bool:
        if market_id not in self.recently_traded:
            return False
        elapsed = (datetime.now() - self.recently_traded[market_id]).total_seconds()
        return elapsed < TRADE_COOLDOWN_SECONDS
    
    def _scan_all_strategies(self) -> List[Opportunity]:
        """Scan ALL strategies and return sorted opportunities"""
        opportunities = []
        
        # Clean old cooldowns
        now = datetime.now()
        self.recently_traded = {
            k: v for k, v in self.recently_traded.items()
            if (now - v).total_seconds() < TRADE_COOLDOWN_SECONDS
        }
        
        for market in self.markets:
            if self._is_on_cooldown(market.id):
                continue
            
            # Strategy 1: Dutch Book (GUARANTEED PROFIT)
            opp = self._check_dutch_book(market)
            if opp:
                opportunities.append(opp)
                continue
            
            # Strategy 2: High-Confidence Snipe
            opp = self._check_snipe(market)
            if opp:
                opportunities.append(opp)
            
            # Strategy 3: Momentum
            opp = self._check_momentum(market)
            if opp:
                opportunities.append(opp)
            
            # Strategy 4: Latency Arb (crypto markets)
            opp = self._check_latency_arb(market)
            if opp:
                opportunities.append(opp)
            
            # Strategy 5: Value Betting
            opp = self._check_value_bet(market)
            if opp:
                opportunities.append(opp)
        
        self.stats.opportunities_found += len(opportunities)
        
        # Sort by expected value (profit * confidence)
        opportunities.sort(
            key=lambda x: x.expected_profit_pct * x.confidence,
            reverse=True
        )
        
        return opportunities
    
    def _check_dutch_book(self, market: Market) -> Optional[Opportunity]:
        """Dutch Book: YES + NO < $1 = guaranteed profit"""
        total = market.yes_price + market.no_price
        
        if total >= self.DUTCH_BOOK_THRESHOLD:
            return None
        
        profit_pct = (1 - total) * 100
        
        return Opportunity(
            market=market,
            strategy="DUTCH_BOOK",
            direction="BOTH",
            expected_profit_pct=profit_pct,
            confidence=0.99,
            details=f"YES={market.yes_price:.3f}+NO={market.no_price:.3f}={total:.3f}",
            suggested_size=self._calculate_order_size(0.99),
        )
    
    def _check_snipe(self, market: Market) -> Optional[Opportunity]:
        """Snipe near-certain outcomes"""
        if len(market.tokens) < 2:
            return None
        
        for direction, price in [("YES", market.yes_price), ("NO", market.no_price)]:
            if price >= self.SNIPE_THRESHOLD:
                profit_pct = (1 - price) * 100
                return Opportunity(
                    market=market,
                    strategy="SNIPE",
                    direction=direction,
                    expected_profit_pct=profit_pct,
                    confidence=price,
                    details=f"{direction}@{price:.2f} → {profit_pct:.1f}% profit",
                    suggested_size=self._calculate_order_size(price),
                )
        
        return None
    
    def _check_momentum(self, market: Market) -> Optional[Opportunity]:
        """Momentum: follow strong price movements"""
        history = self.price_history.get(market.id, deque())
        if len(history) < 5:
            return None
        
        # Get price change over last 30 seconds
        cutoff = datetime.now() - timedelta(seconds=30)
        old_prices = [p.price for p in history if p.timestamp <= cutoff]
        
        if not old_prices:
            return None
        
        old_price = old_prices[-1]
        current_price = market.yes_price
        change = current_price - old_price
        
        # Strong momentum (price moved significantly)
        if abs(change) < self.MOMENTUM_THRESHOLD * 0.01:
            return None
        
        # Follow the momentum
        if change > 0 and current_price < 0.85:  # Going UP, not too expensive
            return Opportunity(
                market=market,
                strategy="MOMENTUM",
                direction="YES",
                expected_profit_pct=change * 100,
                confidence=0.65,
                details=f"YES trending up +{change:.2f}",
                suggested_size=self._calculate_order_size(0.65),
            )
        elif change < 0 and market.no_price < 0.85:  # Going DOWN
            return Opportunity(
                market=market,
                strategy="MOMENTUM",
                direction="NO",
                expected_profit_pct=abs(change) * 100,
                confidence=0.65,
                details=f"NO trending (YES down {change:.2f})",
                suggested_size=self._calculate_order_size(0.65),
            )
        
        return None
    
    def _check_latency_arb(self, market: Market) -> Optional[Opportunity]:
        """Latency arb: Binance moves before Polymarket"""
        if not WEBSOCKETS_AVAILABLE:
            return None
        
        q = market.question.lower()
        
        asset = None
        for a in ["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp"]:
            if a in q:
                asset = a[:3].upper()
                if asset == "BIT":
                    asset = "BTC"
                elif asset == "ETH":
                    asset = "ETH"
                elif asset == "SOL":
                    asset = "SOL"
                elif asset == "XRP":
                    asset = "XRP"
                break
        
        if not asset:
            return None
        
        # Get Binance momentum
        momentum = self.binance.get_momentum(asset, seconds=30)
        
        if abs(momentum) < self.MOMENTUM_THRESHOLD:
            return None
        
        # Check if Polymarket odds are stale
        odds_diff = abs(market.yes_price - market.no_price)
        if odds_diff > 0.25:  # Already repriced
            return None
        
        # Bet in direction of momentum
        if momentum > 0:  # Price going UP
            direction = "YES" if "up" in q else "NO"
        else:  # Price going DOWN
            direction = "NO" if "up" in q else "YES"
        
        confidence = min(0.85, 0.6 + abs(momentum) * 0.1)
        
        return Opportunity(
            market=market,
            strategy="LATENCY_ARB",
            direction=direction,
            expected_profit_pct=abs(momentum) * 10,
            confidence=confidence,
            details=f"Binance {asset} {momentum:+.2f}%",
            suggested_size=self._calculate_order_size(confidence),
        )
    
    def _check_value_bet(self, market: Market) -> Optional[Opportunity]:
        """Value betting: find mispriced markets"""
        # Skip if market is too close to 50/50
        diff = abs(market.yes_price - market.no_price)
        if diff < 0.10:
            return None
        
        # Look for markets where YES+NO significantly differs from 1
        total = market.yes_price + market.no_price
        edge = abs(1 - total)
        
        if edge < self.VALUE_THRESHOLD:
            return None
        
        # If total < 1, both sides are underpriced (buy cheaper one)
        if total < 1:
            if market.yes_price < market.no_price:
                return Opportunity(
                    market=market,
                    strategy="VALUE",
                    direction="YES",
                    expected_profit_pct=edge * 100,
                    confidence=0.6,
                    details=f"YES underpriced (total={total:.3f})",
                    suggested_size=self._calculate_order_size(0.6),
                )
            else:
                return Opportunity(
                    market=market,
                    strategy="VALUE",
                    direction="NO",
                    expected_profit_pct=edge * 100,
                    confidence=0.6,
                    details=f"NO underpriced (total={total:.3f})",
                    suggested_size=self._calculate_order_size(0.6),
                )
        
        return None
    
    async def _execute(self, opp: Opportunity):
        """Execute a trade"""
        self.stats.trades_attempted += 1
        self.recently_traded[opp.market.id] = datetime.now()
        
        logger.info(f"🎯 {opp.strategy}: {opp.market.question[:50]}...")
        logger.info(f"   {opp.direction} | Conf: {opp.confidence:.0%} | Size: ${opp.suggested_size:.2f}")
        logger.info(f"   {opp.details}")
        
        if self.dry_run:
            logger.info(f"   [DRY RUN] Would trade ${opp.suggested_size:.2f}")
            self.stats.trades_successful += 1
            self.stats.total_profit += opp.suggested_size * opp.expected_profit_pct / 100
            self.stats.total_volume += opp.suggested_size
            return
        
        try:
            if opp.direction == "BOTH":
                await self._execute_dutch_book(opp)
            else:
                await self._execute_single(opp)
        except Exception as e:
            logger.error(f"   ❌ Trade error: {e}")
            self.stats.trades_failed += 1
            self.daily_loss += 0.1
    
    async def _execute_dutch_book(self, opp: Opportunity):
        """Execute Dutch Book - buy both sides"""
        tokens = opp.market.tokens
        if len(tokens) < 2:
            return
        
        total_cost = opp.market.yes_price + opp.market.no_price
        shares = opp.suggested_size / total_cost
        
        for idx, (token, price) in enumerate([(tokens[0], opp.market.yes_price), (tokens[1], opp.market.no_price)]):
            token_id = token.get("token_id", "")
            order_price = min(0.99, round(price + 0.01, 2))
            order_size = round(shares * price, 2)
            
            order = OrderArgs(
                token_id=token_id,
                side=BUY,
                size=order_size,
                price=order_price,
            )
            signed = self.client.create_order(order)
            result = self.client.post_order(signed, OrderType.GTC)  # GTC not FOK
            
            if not result.get("orderID"):
                logger.error(f"   ❌ Order {idx+1} failed: {result}")
                self.stats.trades_failed += 1
                return
        
        profit = shares * (1 - total_cost)
        logger.info(f"   ✅ Dutch Book complete! Est. profit: ${profit:.4f}")
        self.stats.trades_successful += 1
        self.stats.total_profit += profit
        self.stats.total_volume += opp.suggested_size
    
    async def _execute_single(self, opp: Opportunity):
        """Execute single-side trade"""
        tokens = opp.market.tokens
        if len(tokens) < 2:
            return
        
        idx = 0 if opp.direction == "YES" else 1
        token_id = tokens[idx].get("token_id", "")
        price = opp.market.yes_price if opp.direction == "YES" else opp.market.no_price
        
        order_price = min(0.99, max(0.01, round(price + 0.01, 2)))
        order_size = round(opp.suggested_size, 2)
        
        logger.info(f"   📤 {opp.direction} @ ${order_price:.2f}, size=${order_size:.2f}")
        
        try:
            order = OrderArgs(
                token_id=token_id,
                side=BUY,
                size=order_size,
                price=order_price,
            )
            signed = self.client.create_order(order)
            result = self.client.post_order(signed, OrderType.GTC)  # GTC = stays open
            
            if result.get("orderID"):
                logger.info(f"   ✅ Order placed! ID: {result.get('orderID')[:20]}...")
                self.stats.trades_successful += 1
                self.stats.total_volume += opp.suggested_size
            else:
                logger.error(f"   ❌ Order failed: {result}")
                self.stats.trades_failed += 1
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            self.stats.trades_failed += 1
    
    def _print_status(self):
        runtime = datetime.now() - self.stats.start_time
        
        table = Table(title="📊 AGGRESSIVE BOT STATUS", show_header=False)
        table.add_column("", style="cyan")
        table.add_column("")
        
        mode = "[red]🔴 LIVE[/red]" if not self.dry_run else "[yellow]DRY RUN[/yellow]"
        
        table.add_row("Mode", mode)
        table.add_row("Runtime", str(runtime).split('.')[0])
        table.add_row("Markets", f"[green]{len(self.markets)}[/green]")
        table.add_row("Scans", f"{self.stats.scans:,}")
        table.add_row("Opportunities", f"[cyan]{self.stats.opportunities_found}[/cyan]")
        table.add_row("Trades", f"[green]{self.stats.trades_successful}[/green]/[red]{self.stats.trades_failed}[/red]")
        table.add_row("Volume", f"${self.stats.total_volume:.2f}")
        table.add_row("Est. Profit", f"[green]${self.stats.total_profit:.4f}[/green]")
        table.add_row("Daily Loss", f"[red]${self.daily_loss:.2f}[/red]")
        
        if WEBSOCKETS_AVAILABLE and self.binance.prices["BTC"] > 0:
            table.add_row("BTC", f"${self.binance.prices['BTC']:,.0f}")
        
        console.print(table)
    
    def _print_final(self):
        console.print(Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"Runtime: {datetime.now() - self.stats.start_time}\n"
            f"Markets Scanned: {len(self.markets)}\n"
            f"Opportunities Found: {self.stats.opportunities_found}\n"
            f"Trades: {self.stats.trades_successful}/{self.stats.trades_attempted}\n"
            f"Volume: ${self.stats.total_volume:.2f}\n"
            f"Est. Profit: ${self.stats.total_profit:.4f}",
            title="📈 Final Stats",
            border_style="green",
        ))


async def main():
    console.print("""
[bold red]
╔══════════════════════════════════════════════════════════════════════╗
║   🚀  ULTIMATE POLYMARKET BOT v3 - AGGRESSIVE MODE  🚀               ║
║                                                                      ║
║   Strategies: Dutch Book | Snipe | Momentum | Value | Latency        ║
║   Risk: 10% per trade | Scan: 300ms | Markets: Quick (<7 days)       ║
╚══════════════════════════════════════════════════════════════════════╝
[/bold red]
    """)
    
    bot = UltimateBot()
    signal.signal(signal.SIGINT, lambda s, f: bot.stop())
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
