"""
auth stuff using supabase
handles login, signup, and session management
"""

import streamlit as st
import os
from typing import Optional, Dict, Any

# try to import supabase, but don't crash if it's not installed
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None


def get_supabase_client() -> Optional[Client]:
    """connects to supabase using credentials from secrets or env vars"""
    if not SUPABASE_AVAILABLE:
        return None

    try:
        # check streamlit secrets first (for deployed apps)
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")

        # fallback to env variables for local dev
        if not url or not key:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

        # no credentials = no auth features
        if not url or not key:
            return None

        return create_client(url, key)

    except Exception as e:
        st.error(f"Failed to connect to authentication service: {e}")
        return None


def init_session_state():
    """sets up session state vars for auth"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'user' not in st.session_state:
        st.session_state.user = None

    if 'auth_error' not in st.session_state:
        st.session_state.auth_error = None


def is_authenticated() -> bool:
    """checks if user is logged in"""
    return st.session_state.get('authenticated', False)


def get_current_user() -> Optional[Dict[str, Any]]:
    """returns current user info if logged in"""
    return st.session_state.get('user', None)


def sign_up(email: str, password: str) -> tuple[bool, str]:
    """creates a new user account"""
    client = get_supabase_client()

    if not client:
        return False, "Authentication service not configured"

    try:
        response = client.auth.sign_up({
            "email": email,
            "password": password
        })

        if response.user:
            return True, f"Account created! Check {email} for verification link."
        else:
            return False, "Signup failed. Email may already be registered."

    except Exception as e:
        return False, f"Signup error: {str(e)}"


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """logs in a user"""
    client = get_supabase_client()

    if not client:
        return False, "Authentication service not configured"

    try:
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if response.user:
            st.session_state.authenticated = True
            st.session_state.user = {
                "id": response.user.id,
                "email": response.user.email
            }
            return True, "Login successful!"
        else:
            return False, "Invalid email or password"

    except Exception as e:
        return False, f"Login error: {str(e)}"


def sign_out():
    """logs out the current user"""
    client = get_supabase_client()

    if client:
        try:
            client.auth.sign_out()
        except:
            pass  # clear local state even if supabase call fails

    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_error = None


def reset_password(email: str) -> tuple[bool, str]:
    """sends password reset email"""
    client = get_supabase_client()

    if not client:
        return False, "Authentication service not configured"

    try:
        client.auth.reset_password_for_email(email)
        return True, f"Password reset email sent to {email}"

    except Exception as e:
        return False, f"Reset error: {str(e)}"


def render_auth_ui():
    """shows login/signup/reset forms"""
    st.markdown("### Authentication")

    # tabs for different auth actions
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset Password"])

    # login tab
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Sign In", use_container_width=True)

            if submit:
                if email and password:
                    success, message = sign_in(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both email and password")

    # signup tab
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
            submit = st.form_submit_button("Create Account", use_container_width=True)

            if submit:
                if email and password and password_confirm:
                    if password == password_confirm:
                        if len(password) >= 6:
                            success, message = sign_up(email, password)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.warning("Password must be at least 6 characters")
                    else:
                        st.warning("Passwords do not match")
                else:
                    st.warning("Please fill in all fields")

    # password reset tab
    with tab3:
        with st.form("reset_form"):
            email = st.text_input("Email", key="reset_email")
            submit = st.form_submit_button("Send Reset Link", use_container_width=True)

            if submit:
                if email:
                    success, message = reset_password(email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter your email")


def render_user_menu():
    """shows user email and sign out button"""
    if is_authenticated():
        user = get_current_user()
        if user:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Logged in as:**")
            st.sidebar.markdown(f"`{user['email']}`")

            if st.sidebar.button("Sign Out", use_container_width=True):
                sign_out()
                st.rerun()


if __name__ == "__main__":
    """test auth module standalone"""
    st.title("Authentication Module Test")

    init_session_state()

    if not is_authenticated():
        render_auth_ui()
    else:
        user = get_current_user()
        st.success(f"Welcome, {user['email']}!")
        render_user_menu()
