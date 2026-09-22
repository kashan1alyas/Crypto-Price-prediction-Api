import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
import time
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict, Any, List
from config import settings, TIMEFRAME_MAP, ALPHA_VANTAGE_API_KEY
from functools import wraps
import ta
from cachetools import TTLCache
import asyncio
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

# Set of major cryptocurrency symbols
CRYPTO_SYMBOLS = {
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'DOT', 'MATIC',
    'LINK', 'UNI', 'LTC', 'ATOM', 'ETC', 'XLM', 'ALGO', 'XMR', 'FIL',
    'TRX', 'AVAX', 'NEAR', 'FTM', 'APE', 'ARB', 'OP',
    'INJ', 'SUI', 'APT', 'SEI', 'TIA', 'JUP', 'WIF', 'BONK', 'PEPE',
    'SHIB', 'SAND', 'MANA', 'AXS', 'GALA', 'IMX', 'CRV', 'LDO', 'RPL',
    'MKR', 'AAVE', 'COMP', 'SNX', 'YFI', 'SUSHI', '1INCH', 'RUNE'
}

# CoinGecko ID mapping for cryptocurrencies
COINGECKO_ID_MAP = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'bnb': 'binancecoin',
    'xrp': 'ripple',
    'ada': 'cardano',
    'doge': 'dogecoin',
    'sol': 'solana',
    'dot': 'polkadot',
    'matic': 'matic-network',
    'link': 'chainlink',
    'uni': 'uniswap',
    'ltc': 'litecoin',
    'atom': 'cosmos',
    'etc': 'ethereum-classic',
    'xlm': 'stellar',
    'algo': 'algorand',
    'xmr': 'monero',
    'fil': 'filecoin',
    'trx': 'tron',
    'avax': 'avalanche-2',
    'near': 'near',
    'ftm': 'fantom',
    'ape': 'apecoin',
    'arb': 'arbitrum',
    'op': 'optimism',
    'inj': 'injective-protocol',
    'sui': 'sui',
    'apt': 'aptos',
    'sei': 'sei-network',
    'tia': 'celestia',
    'jup': 'jupiter-exchange-solana',
    'wif': 'dogwifcoin',
    'bonk': 'bonk',
    'pepe': 'pepe',
    'shib': 'shiba-inu',
    'sand': 'the-sandbox',
    'mana': 'decentraland',
    'axs': 'axie-infinity',
    'gala': 'gala',
    'imx': 'immutable-x',
    'crv': 'curve-dao-token',
    'ldo': 'lido-dao',
    'rpl': 'rocket-pool',
    'mkr': 'maker',
    'aave': 'aave',
    'comp': 'compound-governance-token',
    'snx': 'havven',
    'yfi': 'yearn-finance',
    'sushi': 'sushi',
    '1inch': '1inch',
    'rune': 'thorchain'
}

# Yahoo Finance symbol suffix mapping
YAHOO_FINANCE_SUFFIX = {
    # Crypto gets -USD suffix
    'crypto': '-USD',
    # Stocks can optionally have exchange suffixes, but usually not needed
    'stock': ''
}

# Alpha Vantage symbol format mapping
ALPHA_VANTAGE_FORMAT = {
    # Crypto uses plain symbol (e.g., BTC)
    'crypto': lambda s: s.upper(),
    # Stocks use plain symbol (e.g., AAPL)
    'stock': lambda s: s.upper()
}


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol to its base form (uppercase, no suffixes).
    
    Args:
        symbol: Raw symbol input (e.g., 'BTC', 'btc-USD', 'btc_usd')
        
    Returns:
        Normalized base symbol in uppercase (e.g., 'BTC')
    """
    if not symbol:
        return symbol
    # Remove common suffixes and normalize
    normalized = symbol.upper().split('-')[0].split('_')[0]
    return normalized


def get_yf_symbol(symbol: str) -> str:
    """
    Get Yahoo Finance formatted symbol.
    
    Args:
        symbol: Base symbol (e.g., 'BTC', 'AAPL')
        
    Returns:
        Yahoo Finance formatted symbol (e.g., 'BTC-USD' for crypto, 'AAPL' for stock)
    """
    base = normalize_symbol(symbol)
    if is_crypto(base):
        return f"{base}{YAHOO_FINANCE_SUFFIX['crypto']}"
    return f"{base}{YAHOO_FINANCE_SUFFIX['stock']}"


def get_coingecko_id(symbol: str) -> Optional[str]:
    """
    Get CoinGecko ID for a symbol.
    
    Args:
        symbol: Base symbol (e.g., 'BTC', 'TRX')
        
    Returns:
        CoinGecko slug (e.g., 'bitcoin', 'tron') or None if not found
    """
    base = normalize_symbol(symbol).lower()
    coin_id = COINGECKO_ID_MAP.get(base)
    if not coin_id:
        logger.warning(f"No CoinGecko ID mapping found for symbol '{symbol}' (base: '{base}')")
    return coin_id


def get_av_symbol(symbol: str, asset_type: str = 'crypto') -> str:
    """
    Get Alpha Vantage formatted symbol.
    
    Args:
        symbol: Base symbol (e.g., 'BTC', 'AAPL')
        asset_type: 'crypto' or 'stock'
        
    Returns:
        Alpha Vantage formatted symbol
    """
    base = normalize_symbol(symbol)
    formatter = ALPHA_VANTAGE_FORMAT.get(asset_type, ALPHA_VANTAGE_FORMAT['crypto'])
    return formatter(base)


def is_crypto(symbol: str) -> bool:
    """
    Determine if a given symbol represents a cryptocurrency.
    
    Args:
        symbol (str): The trading symbol to check (e.g., 'BTC', 'AAPL')
        
    Returns:
        bool: True if the symbol represents a cryptocurrency, False otherwise
    """
    base_symbol = normalize_symbol(symbol)
    return base_symbol in CRYPTO_SYMBOLS

# Setup logging
logger = logging.getLogger(__name__)

# Ensure this logger's messages are visible, especially if root logger is restrictive
if not logger.handlers:
    handler = logging.StreamHandler() # Log to console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent duplicate logging to root
logger.setLevel(logging.DEBUG) # Set to DEBUG to capture all relevant messages from this module

def rate_limit(max_per_minute: int = settings.MAX_REQUESTS_PER_MINUTE):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = {}
        min_interval = 60.0 / max_per_minute
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if func.__name__ in last_called:
                time_since_last = now - last_called[func.__name__]
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
            result = func(*args, **kwargs)
            last_called[func.__name__] = time.time()
            return result
        return wrapper
    return decorator

class CoinGeckoAPI:
    """Custom CoinGecko API client with robust error handling"""
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.rate_limit_remaining = 10  # Free tier limit
        self.rate_limit_reset = 0
        
        # Use free tier headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    async def get_coin_market_chart_by_id(self, coin_id: str, vs_currency: str, days: int, desired_api_interval: str = 'daily') -> Dict:
        """Get historical market data for a coin. 
           Desired interval can be 'hourly' or 'daily'.
           CoinGecko defaults to hourly for <=90 days, daily otherwise.
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            
            params = {
                'vs_currency': vs_currency,
                'days': str(days) # CoinGecko expects days as string
            }

            # Only specify interval if we want daily AND days > 90, or if we explicitly want daily for short periods.
            # If days <= 90 and desired_api_interval is 'hourly', we omit `interval` for CoinGecko to default to hourly.
            if not (desired_api_interval == 'hourly' and days <= 90):
                params['interval'] = 'daily'
            
            logger.debug(f"CoinGecko request: coin_id={coin_id}, days={days}, effective_api_interval_param={params.get('interval')}")

            await asyncio.sleep(6)  # Free tier rate limit
            
            response = self.session.get(url, params=params, timeout=30)
            
            # Handle specific HTTP status codes
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"CoinGecko rate limit exceeded (429). Retry-After: {retry_after}s. Waiting...")
                await asyncio.sleep(retry_after)
                response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 401:
                logger.warning("CoinGecko API key unauthorized (401). Falling back to simple price endpoint.")
                return await self._get_fallback_data(coin_id, vs_currency, days)
            
            if response.status_code == 404:
                logger.error(f"CoinGecko: Coin '{coin_id}' not found (404). Check if the coin ID is valid.")
                return None
            
            if response.status_code == 403:
                logger.error("CoinGecko: Access forbidden (403). API key may be invalid or rate limited.")
                return None
                
            response.raise_for_status()
            
            # Validate response content
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"CoinGecko: Invalid JSON response for {coin_id}: {str(e)}")
                return None
            
            if not isinstance(data, dict):
                logger.error(f"CoinGecko: Expected dict response, got {type(data).__name__} for {coin_id}")
                return None
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"CoinGecko: Request timed out for {coin_id} after 30s")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"CoinGecko: Connection error for {coin_id}: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed for {coin_id}: {str(e)}")
            return None
            
    async def _get_fallback_data(self, coin_id: str, vs_currency: str, days: int) -> Dict:
        """Fallback to simple price endpoint for free tier"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data and coin_id in data:
                price = data[coin_id][vs_currency]
                market_cap = data[coin_id].get(f'{vs_currency}_market_cap', 0)
                volume = data[coin_id].get(f'{vs_currency}_24h_vol', 0)
                
                # Generate historical-like data
                now = datetime.now(pytz.UTC)
                timestamps = [(now - timedelta(days=i)).timestamp() * 1000 for i in range(min(days, 30))]
                
                return {
                    'prices': [[ts, price] for ts in timestamps],
                    'market_caps': [[ts, market_cap] for ts in timestamps],
                    'total_volumes': [[ts, volume] for ts in timestamps]
                }
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko fallback request failed: {str(e)}")
            return None

class DataFetcher:
    def __init__(self):
        self.coingecko_client = CoinGeckoAPI()
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5 minute cache
        self.alpha_vantage_ts_client = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        self.alpha_vantage_crypto_client = CryptoCurrencies(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        
    async def get_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical data, first trying CoinGecko and falling back to Yahoo Finance.
        Applies technical indicators and basic cleaning.
        """
        logger.info(f"Fetching historical data for {symbol}, timeframe {timeframe}")
        try:
            df = await self.get_merged_data(symbol, timeframe)

            if df is None or df.empty:
                logger.warning(f"No data returned from get_merged_data for {symbol} ({timeframe}) an empty DataFrame will be returned.")
                return pd.DataFrame()

            min_points = self._get_min_points(symbol, timeframe)
            if len(df) < min_points:
                logger.warning(f"Insufficient data after merging for {symbol} ({timeframe}): got {len(df)}, need {min_points}. An empty DataFrame will be returned.")
                return pd.DataFrame()

            # Add technical indicators
            df_with_indicators, actual_feature_names = self._add_technical_indicators(df.copy(), timeframe)
            
            if df_with_indicators is None or df_with_indicators.empty:
                logger.warning(f"Technical indicators returned empty/None for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Validate and clean final data
            final_df = self._clean_data(df_with_indicators.copy())
            
            if final_df is None or final_df.empty:
                logger.warning(f"Clean data returned empty for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error in get_historical_data: {str(e)}")
            raise

    async def _fetch_coingecko_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data from CoinGecko with retry mechanism"""
        coin_id = None
        for attempt in range(max_retries):
            try:
                coin_id = self._get_coingecko_id(symbol.lower())
                if not coin_id:
                    logger.warning(f"No CoinGecko ID mapping for {symbol}. Skipping CoinGecko fetch.")
                    return None
                
                # Determine desired API interval based on our system's timeframe
                desired_api_interval = 'hourly' if timeframe in ["30m", "1h", "4h"] else 'daily'

                # Calculate days needed
                if desired_api_interval == 'hourly':
                    days_to_request_cg = min(self._calculate_days_needed(timeframe, min_points, source_hint='coingecko_hourly'), 90)
                else:
                    days_to_request_cg = self._calculate_days_needed(timeframe, min_points, source_hint='coingecko_daily')
                    days_to_request_cg = min(days_to_request_cg, 365)

                logger.debug(f"CoinGecko fetch: symbol={symbol}, coin_id={coin_id}, timeframe={timeframe}, desired_api_interval={desired_api_interval}, days_to_request_cg={days_to_request_cg}, min_points_system={min_points}")

                data = await self.coingecko_client.get_coin_market_chart_by_id(
                    coin_id,
                    'usd',
                    days=days_to_request_cg,
                    desired_api_interval=desired_api_interval
                )
                
                # Detailed response logging
                if data is None:
                    logger.warning(f"CoinGecko returned None for {symbol} (coin_id={coin_id}) on attempt {attempt + 1}")
                elif not isinstance(data, dict):
                    logger.warning(f"CoinGecko returned non-dict response for {symbol} (coin_id={coin_id}): type={type(data).__name__}, attempt={attempt + 1}")
                elif 'prices' not in data:
                    logger.warning(f"CoinGecko response missing 'prices' key for {symbol} (coin_id={coin_id}). Keys present: {list(data.keys())}, attempt={attempt + 1}")
                elif data['prices'] is None or len(data['prices']) == 0:
                    logger.warning(f"CoinGecko returned empty 'prices' array for {symbol} (coin_id={coin_id}), attempt={attempt + 1}")
                else:
                    logger.info(f"CoinGecko returned {len(data['prices'])} price points for {symbol} (coin_id={coin_id}), attempt={attempt + 1}")

                if data and 'prices' in data and data['prices']:
                    df = self._process_coingecko_data(data, timeframe)
                    if df is not None and not df.empty and len(df) >= min_points:
                        return df
                    elif df is not None:
                        logger.warning(f"CoinGecko data insufficient for {symbol}: got {len(df)} rows after processing, need {min_points} (attempt {attempt + 1})")
                    else:
                        logger.warning(f"CoinGecko data processing returned None/empty for {symbol} (coin_id={coin_id}), attempt={attempt + 1}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"CoinGecko fetch attempt {attempt + 1} failed for {symbol} (coin_id={coin_id}): {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        logger.error(f"CoinGecko fetch definitively failed for {symbol} (coin_id={coin_id}) after {max_retries} attempts")
        return None

    def _calculate_days_needed(self, timeframe: str, min_points: int, source_hint: Optional[str] = None) -> int:
        """Calculate days needed based on timeframe and minimum points.
           source_hint can be used if calculation differs by source (e.g. 'coingecko_hourly')
        """
        # Default points per day for each timeframe
        # For hourly sources, we might need to adjust this if min_points refers to a specific interval
        if timeframe == "1h": points_per_day_for_timeframe = 24
        elif timeframe == "4h": points_per_day_for_timeframe = 6
        elif timeframe == "30m": points_per_day_for_timeframe = 48 # Assuming 30min resolution
        elif timeframe == "24h": points_per_day_for_timeframe = 1
        elif timeframe == "7d": points_per_day_for_timeframe = 1/7.0
        else: points_per_day_for_timeframe = 1 # Default to daily if unknown

        if points_per_day_for_timeframe == 0: calculated_days = min_points # Avoid division by zero for very coarse timeframes
        else: calculated_days = (min_points / points_per_day_for_timeframe)

        # CoinGecko free API provides max 90 days of hourly data.
        # If we want hourly data from coingecko, and min_points requires more than 90 days of hourly,
        # then coingecko cannot satisfy it with hourly data.
        # This logic should ideally be in _fetch_coingecko_with_retry to decide if it CAN fetch what's needed.
        # Here, we just calculate days based on points and interval implied by timeframe.

        # Ensure a minimum number of days, e.g., CoinGecko might need at least 1 day.
        # And the free tier was previously capped at 30 days as a minimum for demo API in old code.
        # For now, let's ensure at least a few days if calculated is too low.
        final_days = max(2, int(round(calculated_days))) # Request at least 2 days of data
        
        # The old CoinGeckoAPI specific cap `min(days, 90)` is now handled by the caller or CoinGeckoAPI itself
        # based on desired_api_interval. Max 30 used to be for demo API.
        # The `max(30, ...)` in old code was probably to ensure enough span for daily data on free tier.
        # If we are fetching hourly, we may not want to force 30 days if fewer are needed and available.
        # If we are fetching daily, 30 days might be a good minimum if min_points is low.
        if source_hint == 'coingecko_daily': # Check only source_hint
             final_days = max(30, final_days) # If daily for coingecko, ensure at least 30 days span.

        logger.debug(f"_calculate_days_needed for {timeframe}, {min_points}pts, source_hint='{source_hint}': pts_per_day={points_per_day_for_timeframe:.2f}, calc_days={calculated_days:.2f}, final_days={final_days}")
        return final_days

    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Map symbol to CoinGecko ID - delegates to centralized utility"""
        return get_coingecko_id(symbol)

    async def _fetch_yfinance_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with retry and specific error handling"""
        logger.debug(f"Attempting to fetch yfinance data for {symbol}, timeframe {timeframe}, requiring {min_points} points.")
        
        # Ensure crypto symbols are normalized to {SYMBOL}-USD format
        base = normalize_symbol(symbol)
        if is_crypto(base):
            yf_symbol = f"{base}-USD"
        else:
            yf_symbol = symbol.upper()
        
        for attempt in range(max_retries):
            try:
                period = self._get_yfinance_period(timeframe, min_points)
                interval = self._get_yfinance_interval(timeframe)
                
                logger.debug(f"Yfinance request: symbol={yf_symbol}, period={period}, interval={interval}, attempt={attempt + 1}")
                
                ticker = yf.Ticker(yf_symbol)
                data_hist = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    timeout=30
                )
                
                # Detailed response logging
                if data_hist is None:
                    logger.warning(f"YFinance returned None for {yf_symbol} on attempt {attempt + 1}")
                elif data_hist.empty:
                    logger.warning(f"YFinance returned empty DataFrame for {yf_symbol} with period={period}, interval={interval} (attempt {attempt + 1}). Possible causes: invalid symbol, delisted asset, or API restriction.")
                else:
                    logger.info(f"YFinance returned {len(data_hist)} raw data points for {yf_symbol} (attempt {attempt + 1})")
                    logger.info(f"Data range: {data_hist.index.min()} to {data_hist.index.max()}")
                
                if not data_hist.empty:
                    processed_data = self._process_yfinance_data(data_hist.copy())
                    
                    if processed_data is not None and not processed_data.empty:
                        logger.info(f"Processed YFinance data for {symbol}: {len(processed_data)} points")
                        logger.info(f"Processed data range: {processed_data.index.min()} to {processed_data.index.max()}")
                        if len(processed_data) >= min_points:
                            logger.info(f"Sufficient YFinance data obtained for {symbol} ({len(processed_data)} >= {min_points}).")
                            return processed_data
                        else:
                            logger.warning(f"Insufficient YFinance data for {symbol} after processing: got {len(processed_data)}, require {min_points}. Retrying if possible.")
                    else:
                        logger.warning(f"YFinance data for {symbol} became empty or None after processing. Retrying if possible.")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + 1)
                
            except KeyError as e:
                logger.error(f"YFinance: Missing expected data field for {yf_symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt + 1)
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    logger.warning(f"YFinance: Rate limited for {yf_symbol}. Waiting 60s before retry...")
                    await asyncio.sleep(60)
                elif 'no data found' in error_msg or 'invalid period' in error_msg:
                    logger.error(f"YFinance: No data available for {yf_symbol} with period={period}, interval={interval}. Symbol may not support this timeframe.")
                    return None  # Don't retry for invalid period/interval
                else:
                    logger.error(f"YFinance fetch attempt {attempt + 1} for {yf_symbol} failed: {str(e)}", exc_info=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt + 2)
                    
        logger.error(f"YFinance fetch definitively failed for {yf_symbol} after {max_retries} attempts.")
        return None

    async def _fetch_binance_klines(self, symbol: str, timeframe: str, min_points: int) -> Optional[pd.DataFrame]:
        """Fetch sub-hourly crypto data from Binance Public Klines API.
        
        Binance provides up to 1000 candles per request for 30m intervals,
        making it ideal for sub-hourly data that CoinGecko cannot provide
        (CoinGecko caps 30m at ~96 points / 2 days).
        
        Args:
            symbol: Base symbol (e.g., 'BTC', 'SOL')
            timeframe: Timeframe string (e.g., '30m')
            min_points: Minimum number of data points required
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        if timeframe not in ('30m', '1h', '4h'):
            logger.debug(f"Binance Klines only supports sub-hourly intervals, skipping for {timeframe}")
            return None
            
        base = normalize_symbol(symbol)
        binance_symbol = f"{base}USDT"
        
        # Map timeframe to Binance interval
        interval_map = {
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
        }
        binance_interval = interval_map.get(timeframe)
        if not binance_interval:
            logger.warning(f"Unsupported Binance interval for timeframe: {timeframe}")
            return None
        
        try:
            # Binance public API - no auth needed for klines
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': binance_interval,
                'limit': min(1000, max(min_points + 100, 500))  # Request extra for warmup
            }
            
            logger.info(f"Fetching Binance Klines: {binance_symbol} {binance_interval} (limit={params['limit']})")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=30)
            )
            
            if response.status_code != 200:
                logger.warning(f"Binance API returned status {response.status_code} for {binance_symbol}: {response.text[:200]}")
                return None
            
            data = response.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"Binance returned empty data for {binance_symbol}")
                return None
            
            logger.info(f"Binance returned {len(data)} candles for {binance_symbol}")
            
            # Parse Binance klines format:
            # [open_time, open, high, low, close, volume, close_time, ...]
            df = pd.DataFrame(data, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert timestamps to datetime
            df['Timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('Timestamp', inplace=True)
            
            # Convert numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            
            logger.info(f"Processed Binance data for {symbol}: {len(df)} points, range: {df.index.min()} to {df.index.max()}")
            
            if len(df) >= min_points:
                return df
            else:
                logger.warning(f"Binance data insufficient for {symbol}: got {len(df)}, need {min_points}")
                return df  # Return partial data for fallback consideration
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request failed for {binance_symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Binance Klines fetch failed for {binance_symbol}: {str(e)}", exc_info=True)
            return None

    def _process_yfinance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty:
                logger.warning("YFinance: _process_yfinance_data received an empty DataFrame.")
                return pd.DataFrame()
            
            logger.debug(f"YFinance: Initial processing of {len(df)} rows. Columns: {df.columns.tolist()}")
            original_count = len(df)

            # Ensure DateTimeIndex and UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            df.sort_index(inplace=True)

            # Standardize column names
            rename_map = {}
            for col_name in df.columns:
                col_lower = col_name.lower()
                if col_lower == 'open': rename_map[col_name] = 'Open'
                elif col_lower == 'high': rename_map[col_name] = 'High'
                elif col_lower == 'low': rename_map[col_name] = 'Low'
                elif col_lower == 'close': rename_map[col_name] = 'Close'
                elif col_lower == 'adj close': rename_map[col_name] = 'Adj Close'
                elif col_lower == 'volume': rename_map[col_name] = 'Volume'
            df.rename(columns=rename_map, inplace=True)

            # Prioritize 'Adj Close' if available
            if 'Adj Close' in df.columns and not df['Adj Close'].isnull().all():
                logger.debug("YFinance: Using 'Adj Close' as 'Close'.")
                df['Close'] = df['Adj Close']
            elif 'Close' not in df.columns or df['Close'].isnull().all():
                logger.error("YFinance: 'Close' column is missing or all NaN, cannot proceed.")
                return pd.DataFrame()

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_req_cols = [col for col in required_cols if col not in df.columns]
            if missing_req_cols:
                logger.error(f"YFinance data missing required columns after renaming: {missing_req_cols}. Has: {df.columns.tolist()}")
                return pd.DataFrame()

            # Handle NaNs in OHLCV
            df.dropna(subset=['Close'], inplace=True)
            if df.empty:
                logger.warning("YFinance: Data empty after dropping rows with NaN 'Close'.")
                return pd.DataFrame()
            
            # Fill missing values with forward and backward fill, then with 'Close' if still missing
            for col in ['Open', 'High', 'Low']:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill().fillna(df['Close'])
            
            df = df.ffill().bfill()
            
            df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
            df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

            zero_volume_count = (df['Volume'] <= 0).sum()
            if zero_volume_count > 0:
                logger.debug(f"YFinance: Found {zero_volume_count} rows with Volume <= 0.")
                if zero_volume_count < len(df) * 0.5:
                    df = df[df['Volume'] > 0]
                    logger.debug(f"YFinance: Removed zero volume rows. Count after: {len(df)}")
                else:
                    logger.warning(f"YFinance: More than 50% of rows have zero volume ({zero_volume_count}/{len(df)}). Not filtering by volume.")
            
            if df.empty:
                logger.warning("YFinance: Data became empty after processing (NaN drop or zero volume filter).")
                return pd.DataFrame()
            
            df = df[required_cols]

            logger.info(f"YFinance: Processed {original_count} raw rows into {len(df)} cleaned rows.")
            return df
        except Exception as e_process:
            logger.error(f"YFinance: Unhandled exception in _process_yfinance_data: {str(e_process)}", exc_info=True)
            return pd.DataFrame()

    def _process_coingecko_data(self, data: Dict[str, Any], timeframe: str) -> Optional[pd.DataFrame]:
        """Process CoinGecko data to standardized OHLCV format."""
        try:
            if not data or not isinstance(data, dict):
                logger.error("CoinGecko: Invalid data format received")
                return None
            
            # Extract price and volume data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                logger.error("CoinGecko: No price data in response")
                return None
            
            # Create DataFrame with timestamps
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Add volume data if available
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms', utc=True)
                volume_df.set_index('timestamp', inplace=True)
                df['Volume'] = volume_df['Volume']
            else:
                df['Volume'] = 0
            
            # Generate OHLC from close prices
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close'].rolling(2, min_periods=1).max()
            df['Low'] = df['Close'].rolling(2, min_periods=1).min()
            
            # Handle initial NaNs from shifts or missing volume
            df = df.ffill().bfill()

            # Ensure required columns exist after processing
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"CoinGecko: Processed data missing required columns: {missing_cols}. Has: {df.columns.tolist()}")
                # Attempt to fill missing essential OHLC with Close, Volume with 0
                for col in missing_cols:
                    if col in ['Open', 'High', 'Low']:
                        df[col] = df['Close']
                    elif col == 'Volume':
                        df[col] = 0
                # Recheck
                if not all(col in df.columns for col in required_cols):
                     logger.error(f"CoinGecko: Still missing columns after fallback fill: {[c for c in required_cols if c not in df.columns]}. Returning None.")
                     return None
            
            df = df[required_cols]

            # Remove rows where essential data might still be NaN
            df.dropna(subset=['Close', 'Volume'], inplace=True)
            df = df[df['Volume'] >= 0]
            
            if df.empty:
                logger.warning("CoinGecko: Data empty after processing")
                return None
            
            logger.info(f"Successfully processed CoinGecko data. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing CoinGecko data: {str(e)}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> tuple:
        """Add technical indicators to DataFrame"""
        try:
            if df is None or df.empty:
                logger.warning("Cannot add indicators to empty DataFrame")
                return None, None

            if len(df) < 15:
                logger.warning(f"DataFrame too small for reliable indicators ({len(df)} rows); returning as-is")
                return df, list(df.columns)

            df_ta = df.copy()
            eps = 1e-8

            # RSI
            df_ta['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

            # Momentum (safe pct_change via replace)
            shifted = df['Close'].shift(10)
            df_ta['Momentum'] = (df['Close'] - shifted) / shifted.replace(0, eps)

            # ADX
            df_ta['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

            # CCI (manual to add epsilon)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(20).mean()
            mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            df_ta['CCI'] = (tp - sma_tp) / (0.015 * mean_dev + eps)

            # Stochastic (manual to add epsilon)
            low_min = df['Low'].rolling(14).min()
            high_max = df['High'].rolling(14).max()
            df_ta['Stoch_%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + eps)
            df_ta['Stoch_%D'] = df_ta['Stoch_%K'].rolling(3).mean()

            # MFI (manual to add epsilon)
            tp_mfi = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp_mfi * df['Volume']
            pos_mf = pd.Series(0.0, index=df.index)
            neg_mf = pd.Series(0.0, index=df.index)
            tp_diff = tp_mfi.diff()
            pos_mf[tp_diff > 0] = mf[tp_diff > 0]
            neg_mf[tp_diff < 0] = mf[tp_diff < 0]
            pos_sum = pos_mf.rolling(14).sum()
            neg_sum = neg_mf.rolling(14).sum()
            df_ta['MFI'] = 100 - (100 / (1 + pos_sum / (neg_sum + eps)))

            # ATR
            df_ta['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

            # Volatility (relative) with epsilon
            vol = df['Close'].rolling(window=20).std()
            vol_mean = df['Close'].rolling(window=20).mean()
            df_ta['Volatility'] = vol / (vol_mean.abs() + eps)

            # OBV
            df_ta['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

            # VWAP with epsilon
            vwap_num = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
            vwap_den = df['Volume'].cumsum()
            df_ta['VWAP'] = vwap_num / (vwap_den + eps)

            # Volume profile with epsilon
            vol_ma = df['Volume'].rolling(window=20).mean()
            df_ta['Volume_Profile'] = df['Volume'] / (vol_ma + eps)

            # Multi-timeframe momentum (safe pct_change)
            for period, col in [(5, 'Momentum_5'), (10, 'Momentum_10'), (20, 'Momentum_20')]:
                shifted = df['Close'].shift(period)
                df_ta[col] = (df['Close'] - shifted) / shifted.replace(0, eps)

            # Market regime detection (SMA crossover)
            sma_short = df['Close'].rolling(10).mean()
            sma_long = df['Close'].rolling(30).mean()
            df_ta['Market_Regime'] = np.where(sma_short > sma_long, 1.0,
                                     np.where(sma_short < sma_long, -1.0, 0.0))

            # Volatility regime
            ret_vol = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1).replace(0, eps))
            ret_vol = ret_vol.rolling(20).std()
            vol_median = ret_vol.rolling(50).median()
            df_ta['Volatility_Regime'] = np.where(ret_vol > vol_median, 1.0, -1.0)

            # Time features (cyclical encoding)
            if hasattr(df.index, 'hour'):
                df_ta['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                df_ta['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
                df_ta['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                df_ta['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            else:
                df_ta['Hour_Sin'] = 0.0
                df_ta['Hour_Cos'] = 1.0
                df_ta['DayOfWeek_Sin'] = 0.0
                df_ta['DayOfWeek_Cos'] = 1.0
            
            # Handle missing values in technical indicators
            df_ta = df_ta.replace([np.inf, -np.inf], np.nan)
            df_ta = df_ta.ffill().bfill().fillna(0.0)
            
            # Add technical indicators to main dataframe
            df = df.copy()
            for col in df_ta.columns:
                df[col] = df_ta[col]
            
            # Forward fill any remaining NaN values
            df = df.ffill().bfill()
            
            # Get the list of feature columns in the correct order
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns.extend([col for col in df_ta.columns if col not in feature_columns])
            
            # Verify all features are present
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                logger.error(f"Missing features in technical indicators: {missing_features}")
                return None, None
            
            logger.info(f"Generated {len(feature_columns)} technical indicators for {timeframe} timeframe")
            return df[feature_columns], feature_columns
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return None, None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        try:
            if df is None or df.empty:
                logger.warning("Cannot clean empty DataFrame")
                return df if df is not None else pd.DataFrame()

            # Remove rows with zero volume
            df = df[df['Volume'] > 0].copy()
            
            if df.empty:
                logger.warning("All rows removed by zero-volume filter")
                return pd.DataFrame()
            
            # Forward fill missing values
            df = df.ffill().bfill()
            
            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    @rate_limit()
    def get_coingecko_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data from CoinGecko with improved error handling and rate limiting"""
        try:
            coin_id = self._get_coingecko_id(symbol)
            if not coin_id:
                logger.warning(f"No CoinGecko ID mapping for {symbol}")
                return None

            # Get period from TIMEFRAME_MAP, ensure it doesn't exceed 365 days
            period_str = TIMEFRAME_MAP[timeframe].get('period', '30d')
            days = int(period_str[:-1])
            if days > 365:
                logger.warning(f"Adjusting period from {days} to 365 days due to CoinGecko free tier limitation")
                days = 365

            url = f"{settings.COINGECKO_API_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': str(days)
                # Removed interval parameter as it's restricted in demo API
            }

            self._rate_limit()
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds before retry...")
                self.rate_limit_reset = time.time() + retry_after
                time.sleep(retry_after)
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if not data or 'prices' not in data:
                    logger.error(f"Invalid response format from CoinGecko for {symbol}")
                    return None

                df = self._process_coingecko_data(data, timeframe)
                if df is None:
                    logger.warning(f"Failed to process CoinGecko data for {symbol}")
                    return None
                
                min_required = TIMEFRAME_MAP[timeframe].get('min_samples', settings.MIN_DATA_POINTS)
                if len(df) >= min_required:
                    logger.info(f"Successfully fetched and processed CoinGecko data for {symbol}, {len(df)} points.")
                    return df
                else:
                    logger.warning(f"Insufficient data points from CoinGecko for {symbol}: got {len(df)}, require {min_required}")
                    return None

            elif response.status_code == 401:
                logger.error(f"CoinGecko API authentication failed: {response.text}")
                return None
            else:
                logger.error(f"CoinGecko API request failed with status {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching CoinGecko data for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching CoinGecko data for {symbol}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_coingecko_data for {symbol}: {str(e)}")
            return None

    def fetch_yfinance_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance with improved error handling"""
        try:
            yf_symbol = f"{symbol}-USD"
            
            # Use period and interval from TIMEFRAME_MAP
            period_str = TIMEFRAME_MAP[timeframe].get('period', '365d')
            interval = TIMEFRAME_MAP[timeframe].get('interval', '1h')
            
            logger.debug(f"Requesting {period_str} of data for {symbol} from yfinance with {interval} interval.")
            
            # Convert period string to number of days
            try:
                num_days = int(period_str[:-1])
                if num_days > 365 and interval in ['30m', '1h', '4h']:
                    # yfinance has limitations for intraday data
                    num_days = 365
                    logger.warning(f"Adjusted request period to 365 days due to yfinance limitations for {interval} data")
            except ValueError:
                logger.error(f"Invalid period format in TIMEFRAME_MAP for {timeframe}: {period_str}")
                num_days = 365
            
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=num_days)
            
            # Download data
            data = yf.download(
                tickers=yf_symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False
            )
            
            if data is None or data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return None
            
            logger.info(f"Successfully fetched yfinance data for {symbol}, {len(data)} points before processing.")
            
            # Process data
            processed_df = self._process_yfinance_data(data)
            logger.debug(f"Processed yfinance data, {len(processed_df)} rows")
            
            # Verify minimum samples
            min_required = TIMEFRAME_MAP[timeframe].get('min_samples', settings.MIN_DATA_POINTS)
            if len(processed_df) < min_required:
                logger.warning(f"Insufficient data points from yfinance for {symbol}: got {len(processed_df)}, require {min_required}")
                return None
            
            logger.info(f"Successfully processed yfinance data for {symbol}, {len(processed_df)} points.")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
            return None

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1h data to 4h intervals.
        
        Args:
            df (pd.DataFrame): DataFrame with 1h OHLCV data
            
        Returns:
            pd.DataFrame: Resampled 4h data
        """
        try:
            if df.empty:
                return df
                
            # Ensure index is DateTimeIndex and UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            # Define aggregation rules for OHLCV
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Resample to 4h intervals
            resampled = df.resample('4H').agg(agg_dict)
            
            # Remove any rows where all values are NaN
            resampled.dropna(how='all', inplace=True)
            
            # Forward fill any remaining NaN values
            resampled = resampled.ffill().bfill()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error in _resample_to_4h: {str(e)}")
            return pd.DataFrame()

    async def get_merged_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetches data by trying CoinGecko, then yfinance, then Alpha Vantage.
        Uses different data source priorities based on symbol type (crypto vs stock).

        Fallback behaviour:
          - Each source is tried in order. If a source returns 0 rows or fails to meet the
            minimum point threshold it is explicitly logged and the next source is tried.
          - If no source satisfies the minimum but at least one returned partial data, the
            largest partial result is used (with a warning) so we never silently return an
            empty dataset when data is actually available.
        """
        try:
            logger.info(f"Fetching merged data for {symbol} ({timeframe})")

            # Check for unsupported timeframe combinations
            is_stock = not is_crypto(symbol)
            if is_stock and timeframe in unsupported_timeframes:
                error_msg = unsupported_timeframes[timeframe]
                logger.error(error_msg)
                raise ValueError(error_msg)

            cache_key = f"{symbol}_{timeframe}_merged"
            cached_df = self.cache.get(cache_key)
            if cached_df is not None:
                logger.info(f"Returning cached data for {symbol} ({timeframe})")
                return cached_df.copy()

            min_points = self._get_min_points(symbol, timeframe)
            df = None
            source_used = None

            # Track the best (largest) partial result across all sources in case none
            # meet the minimum requirement but some data is still available.
            best_partial_df = None
            best_partial_source = None
            best_partial_rows = -1

            def _consider(candidate: Optional[pd.DataFrame], source_name: str):
                """Update best_partial if candidate has more rows than current best."""
                nonlocal best_partial_df, best_partial_source, best_partial_rows
                if candidate is not None and len(candidate) > best_partial_rows:
                    best_partial_df = candidate
                    best_partial_source = source_name
                    best_partial_rows = len(candidate)

            if is_crypto(symbol):
                # Alpha Vantage's crypto intraday endpoints require a premium subscription,
                # so for intraday crypto intervals we rely on CoinGecko + Binance + Yahoo Finance only
                # and skip Alpha Vantage entirely. Daily (24h) crypto is still allowed as a
                # last-resort fallback since it is available on the free tier.
                intraday_crypto = timeframe in ('30m', '1h', '4h')
                sub_hourly = timeframe == '30m'  # CoinGecko OHLC caps 30m at ~96 points

                # For sub-hourly (30m), try Binance first since CoinGecko cannot provide enough data
                if sub_hourly:
                    logger.info(f"Sub-hourly crypto ({timeframe}): trying Binance Klines first for {symbol}")
                    df_binance = await self._fetch_binance_klines(symbol, timeframe, min_points)
                    if df_binance is not None and len(df_binance) >= min_points:
                        df = df_binance
                        source_used = "Binance"
                    elif df_binance is not None and len(df_binance) > 0:
                        _consider(df_binance, "Binance")
                        logger.warning(f"Binance insufficient for {symbol} ({timeframe}): got {len(df_binance)}, need {min_points}. Trying CoinGecko/YFinance.")
                    else:
                        logger.warning(f"Binance returned no data for {symbol} ({timeframe}). Trying CoinGecko/YFinance.")

                # Try CoinGecko (works well for 1h and above, limited for 30m)
                if df is None and not sub_hourly:
                    logger.info(f"Attempting to fetch crypto data from CoinGecko for {symbol} ({timeframe})")
                    df_coingecko = await self._fetch_coingecko_with_retry(symbol, timeframe, min_points)
                    if df_coingecko is not None and len(df_coingecko) == 0:
                        logger.warning(f"CoinGecko returned 0 rows for {symbol} ({timeframe}); falling back to next source.")
                    elif df_coingecko is not None and len(df_coingecko) >= min_points:
                        df = df_coingecko
                        source_used = "CoinGecko"
                    else:
                        _consider(df_coingecko, "CoinGecko")
                        logger.warning(f"CoinGecko insufficient for {symbol} ({timeframe}) [rows={None if df_coingecko is None else len(df_coingecko)}]. Trying YFinance.")
                elif df is None and sub_hourly:
                    # For 30m, CoinGecko OHLC is limited, try YFinance directly
                    logger.info(f"Skipping CoinGecko for 30m (limited to ~96 points), trying YFinance for {symbol}")
                
                # Try YFinance if still no data
                if df is None:
                    df_yfinance = await self._fetch_yfinance_with_retry(symbol, timeframe, min_points)
                    if df_yfinance is not None and len(df_yfinance) == 0:
                        logger.warning(f"YFinance returned 0 rows for {symbol} ({timeframe}); falling back to next source.")
                    elif df_yfinance is not None and len(df_yfinance) >= min_points:
                        df = df_yfinance
                        source_used = "YFinance"
                    else:
                        _consider(df_yfinance, "YFinance")
                        if intraday_crypto:
                            logger.warning(
                                f"YFinance insufficient for {symbol} ({timeframe}) [rows={None if df_yfinance is None else len(df_yfinance)}]. "
                                f"Alpha Vantage skipped for intraday crypto (requires premium subscription); "
                                f"no further crypto sources for this interval."
                            )
                        else:
                            logger.warning(f"YFinance insufficient for {symbol} ({timeframe}) [rows={None if df_yfinance is None else len(df_yfinance)}]. Trying Alpha Vantage.")
                            df_alphavantage = await self._fetch_alphavantage_with_retry(symbol, timeframe, min_points)
                            if df_alphavantage is not None and len(df_alphavantage) == 0:
                                logger.warning(f"Alpha Vantage returned 0 rows for {symbol} ({timeframe}); no more sources.")
                            elif df_alphavantage is not None and len(df_alphavantage) >= min_points:
                                df = df_alphavantage
                                source_used = "Alpha Vantage"
                            else:
                                _consider(df_alphavantage, "Alpha Vantage")
            else:
                logger.info(f"Attempting to fetch stock data from YFinance for {symbol} ({timeframe})")
                df_yfinance = await self._fetch_yfinance_with_retry(symbol, timeframe, min_points)
                if df_yfinance is not None and len(df_yfinance) == 0:
                    logger.warning(f"YFinance returned 0 rows for {symbol} ({timeframe}); falling back to next source.")
                elif df_yfinance is not None and len(df_yfinance) >= min_points:
                    df = df_yfinance
                    source_used = "YFinance"
                else:
                    _consider(df_yfinance, "YFinance")
                    logger.warning(f"YFinance insufficient for {symbol} ({timeframe}) [rows={None if df_yfinance is None else len(df_yfinance)}]. Trying Alpha Vantage.")
                    df_alphavantage = await self._fetch_alphavantage_with_retry(symbol, timeframe, min_points)
                    if df_alphavantage is not None and len(df_alphavantage) == 0:
                        logger.warning(f"Alpha Vantage returned 0 rows for {symbol} ({timeframe}); no more sources.")
                    elif df_alphavantage is not None and len(df_alphavantage) >= min_points:
                        df = df_alphavantage
                        source_used = "Alpha Vantage"
                    else:
                        _consider(df_alphavantage, "Alpha Vantage")

            # None of the sources met the minimum requirement: use the best partial result
            # if any data was returned, so we don't silently return an empty dataset.
            if df is None and best_partial_df is not None:
                if best_partial_rows < 30:
                    logger.warning(
                        f"Best partial result too small ({best_partial_rows} rows) for "
                        f"{symbol} ({timeframe}); skipping. Need at least 30 rows for indicators."
                    )
                else:
                    logger.warning(
                        f"No source met minimum requirement ({min_points} pts) for {symbol} ({timeframe}), "
                        f"but using best partial result from {best_partial_source} ({best_partial_rows} rows)."
                    )
                    df = best_partial_df
                    source_used = best_partial_source

            if df is not None:
                df_with_indicators, _ = self._add_technical_indicators(df.copy(), timeframe)
                # If indicator engineering produced an empty dataset, reject it rather than
                # returning 0 rows upstream (which surfaces as a confusing "got 0" error).
                if df_with_indicators is None or len(df_with_indicators) == 0:
                    logger.error(
                        f"Data for {symbol} ({timeframe}) became empty after adding technical "
                        f"indicators; rejecting instead of returning an empty dataset."
                    )
                    df = None
                else:
                    self.cache[cache_key] = df_with_indicators.copy()
                    logger.info(f"Data fetched and processed for {symbol} ({timeframe}) using {source_used}. Shape: {df_with_indicators.shape}")
                    return df_with_indicators

            # Detailed failure logging
            logger.error(f"Failed to fetch sufficient data for {symbol} ({timeframe}) from all sources.")
            logger.error(f"  - Required minimum data points: {min_points}")
            logger.error(f"  - Symbol type: {'crypto' if is_crypto(symbol) else 'stock'}")
            if is_crypto(symbol):
                logger.error(f"  - CoinGecko: Attempted with coin_id={get_coingecko_id(symbol) or 'MAPPING_NOT_FOUND'}")
                logger.error(f"  - YFinance: Attempted with symbol={get_yf_symbol(symbol)}")
                logger.error(f"  - Alpha Vantage: Attempted with symbol={get_av_symbol(symbol, 'crypto')}")
            else:
                logger.error(f"  - YFinance: Attempted with symbol={get_yf_symbol(symbol)}")
                logger.error(f"  - Alpha Vantage: Attempted with symbol={get_av_symbol(symbol, 'stock')}")
            return None
        except Exception as e:
            logger.error(f"Error in get_merged_data for {symbol} ({timeframe}): {str(e)}", exc_info=True)
            return None

    def _get_min_points(self, symbol: str, timeframe: str) -> int:
        """
        Get minimum required data points for a timeframe, adjusted for asset type.
        Stocks have lower requirements due to limited trading hours.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC', 'AAPL')
            timeframe (str): Time interval ('30m', '1h', '4h', '24h')
            
        Returns:
            int: Minimum required data points
        """
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Different requirements based on asset type
        if is_crypto(symbol):
            # Crypto trades 24/7 - keep existing higher requirements
            return TIMEFRAME_MAP[timeframe].get('min_samples', settings.MIN_DATA_POINTS)
        else:
            # Stocks trade ~7 hours/day, 5 days/week
            # Calculate realistic requirements based on trading hours
            stock_requirements = {
                '30m': 336,  # ~30 trading days * 14 periods/day (7 hours)
                '1h': 180,   # ~30 trading days * 7 periods/day
                '4h': 90,    # ~30 trading days * 2 periods/day
                '24h': 60    # ~3 months of trading days
            }
            return stock_requirements.get(timeframe, settings.MIN_DATA_POINTS)
        
    def _get_coingecko_interval(self, timeframe: str) -> Optional[str]:
        """Get CoinGecko interval for timeframe"""
        interval_map = {
            "1h": "hourly",
            "4h": "hourly",
            "24h": "daily",
            "7d": "daily"
        }
        return interval_map.get(timeframe)
        
    def _get_yfinance_interval(self, timeframe: str) -> str:
        """Get yfinance interval for timeframe"""
        interval_map = {
            "30m": "30m",  # Added 30m support
            "1h": "1h",
            "24h": "1d",
            "7d": "1d"
        }
        return interval_map.get(timeframe, "1d")
        
    def _get_yfinance_period(self, timeframe: str, min_points: int) -> str:
        """Get yfinance period based on timeframe and minimum points"""
        # Calculate minimum days needed
        days_map = {
            "30m": min_points / 48,  # 48 points per day for 30m
            "1h": min_points / 24,   # 24 points per day
            "24h": min_points,       # 1 point per day
            "7d": min_points * 7     # 1 point per week
        }
        
        days_needed = int(days_map.get(timeframe, min_points))
        
        # YFinance limitations for intraday data
        if timeframe in ["30m", "1h", "4h"]:
            if timeframe == "30m":
                # For 30m, use 30d period (within 60-day intraday limit)
                # This ensures 300+ candles: 30 days * 48 points/day = 1440 points
                days_needed = 30
                logger.info(f"Requesting {days_needed}d period for 30m data from YFinance (expecting ~{days_needed * 48} points)")
            else:
                # For 1h and 4h, can request more
                days_needed = min(days_needed, 90)
            logger.info(f"Adjusted request period to {days_needed} days due to YFinance intraday limitations for {timeframe}")
        
        # Add a small buffer to the period
        return f"{days_needed + 3}d"

    def _get_alphavantage_symbol(self, symbol: str) -> str:
        """Formats symbol for Alpha Vantage (e.g., BTC -> BTCUSD for stocks/generic, or just BTC for crypto client)."""
        # For TimeSeries client (stocks), format might be just the symbol, e.g. MSFT
        # For CryptoCurrencies client, symbol is like 'BTC', market is 'USD'
        return symbol.upper() # Crypto client takes 'BTC', not 'BTCUSD'

    async def _fetch_alphavantage_with_retry(self, symbol: str, timeframe: str, min_points: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage with retry mechanism, handling both crypto and stocks."""
        logger.debug(f"Attempting to fetch Alpha Vantage data for {symbol}, timeframe {timeframe}, requiring {min_points} points.")
        
        # Determine if this is a crypto or stock symbol
        symbol_is_crypto = is_crypto(symbol)
        symbol_upper = get_av_symbol(symbol, 'crypto' if symbol_is_crypto else 'stock')

        # Alpha Vantage's crypto intraday endpoints require a premium subscription. Skip early
        # for intraday crypto intervals to avoid wasted API calls and confusing failures.
        if symbol_is_crypto and timeframe in ('30m', '1h', '4h'):
            logger.warning(
                f"Alpha Vantage skipped for crypto intraday interval {timeframe} ({symbol}). "
                f"Intraday crypto data requires a premium subscription; use CoinGecko or Yahoo Finance."
            )
            return None
        
        # Initialize appropriate client if not already done
        if symbol_is_crypto:
            if not hasattr(self, 'alpha_vantage_crypto_client'):
                self.alpha_vantage_crypto_client = CryptoCurrencies(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas')
        else:  # Stock
            if not hasattr(self, 'alpha_vantage_stock_client'):
                self.alpha_vantage_stock_client = TimeSeries(key=settings.ALPHA_VANTAGE_API_KEY, output_format='pandas')

        # Map timeframes to Alpha Vantage intervals
        interval_map = {
            '30m': '30min',
            '1h': '60min',
            '24h': 'daily'
        }
        
        av_interval = interval_map.get(timeframe)
        if not av_interval:
            logger.warning(f"Timeframe {timeframe} not supported by Alpha Vantage API")
            return None

        for attempt in range(max_retries):
            try:
                if symbol_is_crypto:
                    # Crypto data fetching
                    logger.debug(f"Alpha Vantage Crypto attempt {attempt+1} for {symbol_upper} (target timeframe: {timeframe})")
                    if timeframe == '24h':
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_crypto_client.get_digital_currency_daily,
                            symbol=symbol_upper,
                            market='USD'
                        )
                    else:
                        # Note: Alpha Vantage crypto API might not support all intraday intervals
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_crypto_client.get_digital_currency_daily,  # Fallback to daily for now
                            symbol=symbol_upper,
                            market='USD'
                        )
                else:
                    # Stock data fetching
                    logger.debug(f"Alpha Vantage Stock attempt {attempt+1} for {symbol_upper} (target timeframe: {timeframe})")
                    if timeframe == '24h':
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_stock_client.get_daily,
                            symbol=symbol_upper,
                            outputsize='full'
                        )
                    else:
                        data, meta_data = await asyncio.to_thread(
                            self.alpha_vantage_stock_client.get_intraday,
                            symbol=symbol_upper,
                            interval=av_interval,
                            outputsize='full'
                        )

                # Validate response
                if data is None:
                    logger.warning(f"Alpha Vantage returned None for {symbol_upper} on attempt {attempt + 1}")
                elif data.empty:
                    logger.warning(f"Alpha Vantage returned empty DataFrame for {symbol_upper} on attempt {attempt + 1}")
                else:
                    logger.info(f"Alpha Vantage returned {len(data)} rows for {symbol_upper} on attempt {attempt + 1}")

                if data is not None and not data.empty:
                    # Process data based on type
                    if symbol_is_crypto:
                        df = self._process_alphavantage_data(data, symbol_upper, 
                                                         target_timeframe=timeframe,
                                                           is_crypto_daily_format=True)
                    else:
                        df = self._process_alphavantage_stock_data(data, symbol_upper, timeframe)
                    
                    if df is not None and len(df) >= min_points:
                        logger.info(f"Successfully fetched {symbol_is_crypto and 'crypto' or 'stock'} data from Alpha Vantage for {symbol_upper}. Points: {len(df)}")
                        return df
                    else:
                        logger.warning(f"Insufficient data from Alpha Vantage for {symbol_upper} after processing: got {len(df) if df is not None else 0}, need {min_points}")
                else:
                    logger.warning(f"No data returned from Alpha Vantage for {symbol_upper} on attempt {attempt + 1}")

                if attempt < max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 15)  # Increased sleep for AV free tier
                
            except ValueError as e:
                error_msg = str(e).lower()
                if 'invalid api call' in error_msg:
                    logger.error(f"Alpha Vantage: Invalid API call for {symbol_upper}. The symbol or interval may not be supported.")
                    return None  # Don't retry invalid API calls
                elif 'note' in error_msg and 'frequency' in error_msg:
                    logger.warning(f"Alpha Vantage: API frequency limit reached for {symbol_upper}. Waiting 60s...")
                    await asyncio.sleep(60)
                else:
                    logger.error(f"Alpha Vantage: Value error for {symbol_upper}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((2 ** attempt) * 15)
            except Exception as e:
                error_msg = str(e).lower()
                if 'api key' in error_msg or 'invalid api' in error_msg:
                    logger.error(f"Alpha Vantage: API key issue or invalid call for {symbol_upper}. Breaking retry loop.")
                    break
                elif 'rate limit' in error_msg or 'too many' in error_msg:
                    logger.warning(f"Alpha Vantage: Rate limited for {symbol_upper}. Waiting 60s...")
                    await asyncio.sleep(60)
                elif 'connection' in error_msg or 'timeout' in error_msg:
                    logger.error(f"Alpha Vantage: Connection/timeout error for {symbol_upper}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((2 ** attempt) * 20)
                else:
                    logger.error(f"Alpha Vantage fetch attempt {attempt + 1} for {symbol_upper} failed: {str(e)}", exc_info=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep((2 ** attempt) * 20)
            
        logger.warning(f"Failed to fetch sufficient data from Alpha Vantage for {symbol_upper} after {max_retries} attempts.")
        return None

    def _process_alphavantage_data(self, df: pd.DataFrame, symbol_orig: str, target_timeframe: str, 
                                   is_crypto_daily_format: bool = False, 
                                   is_intraday_source: bool = False) -> Optional[pd.DataFrame]:
        """Process Alpha Vantage data to match expected format.
           Handles stock format, crypto daily format, and crypto intraday format.
           Resamples intraday data to the target_timeframe.
        """
        try:
            logger.debug(f"Processing Alpha Vantage data for {symbol_orig}. Target timeframe: {target_timeframe}, DailyCryptoFmt: {is_crypto_daily_format}, IntradaySrc: {is_intraday_source}. Initial Cols: {df.columns.tolist()}")
            
            # AV pandas output often has 'date' as index name for daily, or 'index' after reset.
            # Intraday get_digital_currency_intraday directly returns DataFrame with DatetimeIndex.
            if not isinstance(df.index, pd.DatetimeIndex):
                # If 'date' is a column, set it as index. Otherwise, try resetting.
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                else: # This case might happen if index is not named 'date' and not DatetimeIndex
                    df.reset_index(inplace=True) # Try to make 'date' or 'index' a column
                    date_col_found = False
                    for col_candidate in ['date', 'index', 'timestamp']: # common names for date column
                        if col_candidate in df.columns:
                            df.rename(columns={col_candidate: 'Timestamp'}, inplace=True)
                            df.set_index('Timestamp', inplace=True)
                            date_col_found = True
                            break
                    if not date_col_found:
                        logger.error(f"AV: Could not identify a date/timestamp column to set as index. Columns: {df.columns.tolist()}")
                        return None
            
            # Ensure index is DatetimeIndex and UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC') # Assume UTC if not specified by AV
            else:
                df.index = df.index.tz_convert('UTC')
            df.sort_index(inplace=True)

            col_map = {}
            if is_intraday_source:
                # Intraday crypto columns from alpha_vantage lib are: '1. open', '2. high', '3. low', '4. close', '5. volume'
                # This is for client.get_digital_currency_intraday
                for col in df.columns:
                    if '1. open' in col: col_map[col] = 'Open'
                    elif '2. high' in col: col_map[col] = 'High'
                    elif '3. low' in col: col_map[col] = 'Low'
                    elif '4. close' in col: col_map[col] = 'Close'
                    elif '5. volume' in col: col_map[col] = 'Volume'
                df.rename(columns=col_map, inplace=True)

            elif is_crypto_daily_format:
                # Daily Crypto columns: '1a. open (USD)', '2a. high (USD)', etc.
                # Market symbol (e.g. USD) might be needed if not always USD
                market_suffix = "(USD)" # Assuming USD, adjust if market can vary and is in col name
                for col in df.columns:
                    col_lower = col.lower()
                    if f'1a. open {market_suffix.lower()}' in col_lower or (f'1. open {market_suffix.lower()}' in col_lower and "1a." not in col_lower) : col_map[col] = 'Open' # AV sometimes uses 1. open (USD) for daily too
                    elif f'2a. high {market_suffix.lower()}' in col_lower or (f'2. high {market_suffix.lower()}' in col_lower and "2a." not in col_lower): col_map[col] = 'High'
                    elif f'3a. low {market_suffix.lower()}' in col_lower or (f'3. low {market_suffix.lower()}' in col_lower and "3a." not in col_lower): col_map[col] = 'Low'
                    elif f'4a. close {market_suffix.lower()}' in col_lower or (f'4. close {market_suffix.lower()}' in col_lower and "4a." not in col_lower): col_map[col] = 'Close'
                    elif '5. volume' == col_lower and not market_suffix.lower() in col_lower : col_map[col] = 'Volume' # '5. volume'
                    elif f'6. market cap {market_suffix.lower()}' in col_lower : pass # Ignoring market cap for now
                df.rename(columns=col_map, inplace=True)

            else: # Original stock format (e.g., for TimeSeries client if ever used for other assets)
                df.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
                    '4. close': 'Close', '5. adjusted close': 'Adj Close', 
                    '6. volume': 'Volume'
                }, inplace=True)
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']

            # Ensure all essential columns are numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else: # If a required column is missing after mapping, this is an issue
                    logger.error(f"AV: Essential column '{col}' missing after renaming. Available: {df.columns.tolist()}")
                    return None
            
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True) # Drop rows where essential values are NaN
            if df.empty:
                logger.warning(f"AV: Data for {symbol_orig} empty after mapping and NaN drop.")
                return None

            # Resample if intraday source data needs to be converted to a coarser target_timeframe
            # AV intraday is often 1-min or 5-min. Our target_timeframe could be "30m", "1h", "4h".
            if is_intraday_source and target_timeframe in ["30m", "1h", "4h"]:
                logger.debug(f"AV: Resampling intraday data for {symbol_orig} to {target_timeframe}. Original points: {len(df)}")
                
                # Map our timeframe string to pandas offset string
                resample_map = {"30m": "30T", "1h": "1H", "4h": "4H"}
                pandas_timeframe = resample_map.get(target_timeframe)

                if pandas_timeframe:
                    ohlc_dict = {
                        'Open': 'first', 'High': 'max', 'Low': 'min', 
                        'Close': 'last', 'Volume': 'sum'
                    }
                    # Only resample if there are columns to aggregate
                    cols_to_agg = {k: v for k,v in ohlc_dict.items() if k in df.columns}
                    if not cols_to_agg:
                        logger.error(f"AV: No columns to aggregate for resampling {symbol_orig} to {target_timeframe}. Columns: {df.columns.tolist()}")
                        return None

                    df = df.resample(pandas_timeframe).agg(cols_to_agg)
                    df.dropna(how='all', inplace=True) # Remove intervals that are all NaN after resampling
                    # Forward fill is important after resampling to handle periods with no trades
                    # but we also need to be careful not to fill too much into the future if data ends.
                    # df.ffill(inplace=True) # Let's see if this is needed after .agg
                    if df.empty:
                        logger.warning(f"AV: Data for {symbol_orig} empty after resampling to {target_timeframe}.")
                        return None
                    logger.debug(f"AV: Resampled to {len(df)} points for {target_timeframe}.")
                else:
                    logger.warning(f"AV: Could not map target_timeframe '{target_timeframe}' to pandas resample string. No resampling performed.")
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_final_cols = [col for col in required_cols if col not in df.columns]
            if missing_final_cols:
                logger.error(f"AV: Missing expected columns after all processing for {symbol_orig} (target {target_timeframe}): {missing_final_cols}. Available: {df.columns.tolist()}")
                return None

            # Final selection and order
            processed_df = df[required_cols].copy()
            
            # Drop rows with zero volume as a final cleaning step, if not already done
            processed_df = processed_df[processed_df['Volume'] > 0]
            if processed_df.empty:
                logger.warning(f"AV: Data for {symbol_orig} became empty after zero volume filter (target {target_timeframe}).")
                return None

            logger.info(f"Successfully processed Alpha Vantage data for {symbol_orig} (target {target_timeframe}). Shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data for {symbol_orig} (target {target_timeframe}): {str(e)}", exc_info=True)
            return None 

    def _process_alphavantage_stock_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Process stock data from Alpha Vantage TimeSeries client.
        
        Args:
            df (pd.DataFrame): Raw data from Alpha Vantage
            symbol (str): Stock symbol
            timeframe (str): Time interval
            
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame with standardized columns
        """
        try:
            logger.debug(f"Processing Alpha Vantage stock data for {symbol}. Initial shape: {df.shape}")
            
            # Ensure DateTimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Sort by datetime ascending
            df = df.sort_index()
            
            # Rename columns from Alpha Vantage format to our standard format
            rename_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
                '5. adjusted close': 'Adj Close'  # Some endpoints include adjusted close
            }
            df.rename(columns=rename_map, inplace=True)
            
            # If we have adjusted close, use it as the close price
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
                df.drop('Adj Close', axis=1, inplace=True)
            
            # Ensure all required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in Alpha Vantage stock data: {missing_cols}")
                return None
            
            # Convert columns to numeric, replacing any non-numeric values with NaN
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values in essential columns
            df.dropna(subset=['Close'], inplace=True)
            
            # Forward fill remaining NaN values
            df = df.ffill().bfill()
            
            # Remove rows with zero volume
            df = df[df['Volume'] > 0]
            
            if df.empty:
                logger.warning(f"Alpha Vantage stock data for {symbol} became empty after processing")
                return None
            
            # Select only the required columns in the correct order
            df = df[required_cols]
            
            logger.info(f"Successfully processed Alpha Vantage stock data for {symbol}. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage stock data for {symbol}: {str(e)}")
            return None

    async def get_latest_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get the latest market data for a given symbol and timeframe.
        Uses different data source priorities for crypto vs stocks.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC' for Bitcoin, 'AAPL' for Apple)
            timeframe (str): Time interval ('30m', '1h', '4h', '24h')
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with OHLCV data or None if all sources fail
        """
        logger.info(f"Fetching latest data for {symbol} ({timeframe})")
        
        if is_crypto(symbol):
            # For cryptocurrencies: Try CoinGecko -> YFinance -> Alpha Vantage
            logger.debug(f"Treating {symbol} as cryptocurrency")
            
            # Try CoinGecko first for crypto
            try:
                data = await self._fetch_coingecko_data(symbol, timeframe)
                if self._validate_data(data, symbol, timeframe):
                    logger.info(f"Successfully fetched data from CoinGecko for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"CoinGecko failed or returned insufficient data for {symbol} ({timeframe}). Error: {str(e)}")
            
            # Try YFinance with -USD suffix
            try:
                yf_symbol = f"{symbol}-USD"
                data = await self._fetch_yfinance_data(yf_symbol, timeframe)
                if self._validate_data(data, symbol, timeframe):
                    logger.info(f"Successfully fetched data from YFinance for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"YFinance failed or returned insufficient data for {symbol} ({timeframe}). Error: {str(e)}")
            
            # Finally try Alpha Vantage crypto endpoints
            try:
                data = await self._fetch_alphavantage_crypto_data(symbol, timeframe)
                if self._validate_data(data, symbol, timeframe):
                    logger.info(f"Successfully fetched data from Alpha Vantage for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Alpha Vantage failed or returned insufficient data for {symbol} ({timeframe}). Error: {str(e)}")
        
        else:
            # For stocks: Try YFinance -> Alpha Vantage (skip CoinGecko)
            logger.debug(f"Treating {symbol} as stock")
            
            # Try YFinance first for stocks
            try:
                data = await self._fetch_yfinance_data(symbol, timeframe)
                if self._validate_data(data, symbol, timeframe):
                    logger.info(f"Successfully fetched data from YFinance for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"YFinance failed or returned insufficient data for {symbol} ({timeframe}). Error: {str(e)}")
            
            # Try Alpha Vantage stock endpoints
            try:
                data = await self._fetch_alphavantage_stock_data(symbol, timeframe)
                if self._validate_data(data, symbol, timeframe):
                    logger.info(f"Successfully fetched data from Alpha Vantage for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Alpha Vantage failed or returned insufficient data for {symbol} ({timeframe}). Error: {str(e)}")
        
        logger.error(f"Failed to fetch sufficient data for {symbol} ({timeframe}) from all sources.")
        return None

# Define unsupported timeframes for stocks
unsupported_timeframes = {
    '2h': "2-hour timeframe is not supported for stocks with current data providers.",
    '4h': "4-hour timeframe is not supported for stocks with current data providers. Please use 1h or daily timeframes.",
    '6h': "6-hour timeframe is not supported for stocks with current data providers.",
    '8h': "8-hour timeframe is not supported for stocks with current data providers.",
    '12h': "12-hour timeframe is not supported for stocks with current data providers."
}