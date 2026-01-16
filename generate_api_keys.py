#!/usr/bin/env python3
"""
Generate Polymarket API Credentials

This script generates your L2 API credentials (apiKey, secret, passphrase)
from your wallet's private key.

Run this ONCE to get your credentials, then save them to .env
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("🔑 POLYMARKET API KEY GENERATOR")
    print("=" * 60)
    print()
    
    # Check for private key
    private_key = os.getenv("PRIVATE_KEY", "").strip()
    
    if not private_key or private_key == "your_private_key_without_0x_prefix":
        print("❌ ERROR: PRIVATE_KEY not set in .env file!")
        print()
        print("Steps to get your private key from MetaMask:")
        print("  1. Open MetaMask browser extension")
        print("  2. Click the three dots ⋮ → Account Details")
        print("  3. Click 'Show Private Key'")
        print("  4. Enter your password")
        print("  5. Copy the key (WITHOUT the 0x prefix)")
        print("  6. Paste it in your .env file as PRIVATE_KEY=...")
        print()
        sys.exit(1)
    
    # Add 0x prefix if not present
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    print("✓ Private key found")
    print()
    
    try:
        # Import py-clob-client
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        
        print("Connecting to Polymarket CLOB API...")
        print()
        
        # Initialize client with just private key (L1)
        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,  # Polygon mainnet
            key=private_key,
        )
        
        print("✓ Connected to Polymarket")
        print()
        
        # Try to derive existing credentials first
        print("Checking for existing API credentials...")
        try:
            creds = client.derive_api_key()
            print("✓ Found existing API credentials!")
        except Exception:
            # Create new credentials
            print("Creating new API credentials...")
            creds = client.create_api_key()
            print("✓ New API credentials created!")
        
        print()
        print("=" * 60)
        print("🎉 YOUR API CREDENTIALS (save these to .env):")
        print("=" * 60)
        print()
        print(f"POLYMARKET_API_KEY={creds.api_key}")
        print(f"POLYMARKET_API_SECRET={creds.api_secret}")
        print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
        print()
        print("=" * 60)
        print()
        print("⚠️  IMPORTANT:")
        print("  1. Copy these values to your .env file")
        print("  2. NEVER share these credentials")
        print("  3. If you lose them, run this script again")
        print()
        
        # Test the credentials
        print("Testing credentials...")
        try:
            # Create client with full credentials
            test_client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=private_key,
                creds=ApiCreds(
                    api_key=creds.api_key,
                    api_secret=creds.api_secret,
                    api_passphrase=creds.api_passphrase,
                ),
            )
            # Try to get API keys (authenticated endpoint)
            test_client.get_api_keys()
            print("✓ Credentials verified and working!")
        except Exception as e:
            print(f"⚠️  Could not verify credentials: {e}")
            print("   (They may still work - try running the bot)")
        
    except ImportError:
        print("❌ py-clob-client not installed!")
        print()
        print("Run: pip install py-clob-client")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Common issues:")
        print("  - Invalid private key format")
        print("  - Network connection problem")
        print("  - Polymarket API temporarily unavailable")
        sys.exit(1)


if __name__ == "__main__":
    main()
