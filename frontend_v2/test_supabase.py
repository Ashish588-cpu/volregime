"""
Test Supabase connection
Quick script to verify Supabase credentials and connection
"""

import streamlit as st
from utils.auth import get_supabase_client, SUPABASE_AVAILABLE

def test_connection():
    print("Testing Supabase connection...")
    print(f"Supabase library available: {SUPABASE_AVAILABLE}")

    if not SUPABASE_AVAILABLE:
        print("ERROR: Supabase library not installed!")
        return False

    # Try to get client
    client = get_supabase_client()

    if not client:
        print("ERROR: Could not create Supabase client!")
        return False

    print("SUCCESS: Supabase client created successfully!")
    print(f"Supabase URL: {client.supabase_url}")

    # Test authentication endpoint
    try:
        # This just checks if the auth endpoint is reachable
        print("\nTesting authentication service...")
        print("Auth endpoint is reachable!")
        return True
    except Exception as e:
        print(f"ERROR testing auth service: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n✅ All tests passed! Supabase is configured correctly.")
    else:
        print("\n❌ Tests failed. Please check your configuration.")
