"""
Authentication page - Login and Sign Up
Clean dedicated page for user authentication
"""

import streamlit as st
import sys
import os

# need parent dir in path to import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import sign_in, sign_up, reset_password, is_authenticated


def show():
    """shows the authentication page"""

    # Check if Supabase is configured
    from utils.auth import get_supabase_client, SUPABASE_AVAILABLE

    # Only clear auth state on initial page load, not after form submission
    # This prevents clearing auth right after successful login
    if 'auth_page_loaded' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.auth_page_loaded = True

    supabase_configured = False
    if SUPABASE_AVAILABLE:
        client = get_supabase_client()
        if client:
            supabase_configured = True

    # Hero header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d0d10, #1a1a1d);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #00eaff;
        box-shadow: 0 0 30px rgba(0,234,255,0.3);
    ">
        <h1 style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00eaff, #ff2b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 1rem 0;
        ">Welcome to VolRegime</h1>
        <p style="
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0;">
            Sign in to access your portfolio and personalized insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show warning if Supabase not configured
    if not supabase_configured:
        st.warning("""
        **Authentication service is not configured.**

        To enable user authentication, you need to set up Supabase credentials:
        1. Create a free account at [supabase.com](https://supabase.com)
        2. Create a new project
        3. Add your credentials to `.streamlit/secrets.toml` or environment variables:
           - `SUPABASE_URL`
           - `SUPABASE_KEY`

        For now, you can continue as a guest to explore the platform.
        """)

    # Create tabs for different auth actions
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset Password"])

    # Login tab
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: #0d0d10;
                border: 1px solid #00eaff;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 0 20px rgba(0,234,255,0.2);
            ">
                <h3 style="color: #00eaff; text-align: center; margin-bottom: 1.5rem;">Sign In to Your Account</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("login_form"):
                email = st.text_input("Email", placeholder="your.email@example.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submit:
                    if email and password:
                        success, message = sign_in(email, password)
                        if success:
                            st.success(message)
                            # Clear the auth page flag so state resets next time
                            if 'auth_page_loaded' in st.session_state:
                                del st.session_state.auth_page_loaded
                            st.session_state.current_page = "Home"
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter both email and password")

    # Signup tab
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: #0d0d10;
                border: 1px solid #ff2b6b;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 0 20px rgba(255,43,107,0.2);
            ">
                <h3 style="color: #ff2b6b; text-align: center; margin-bottom: 1.5rem;">Create New Account</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("signup_form"):
                email = st.text_input("Email", placeholder="your.email@example.com", key="signup_email")
                password = st.text_input("Password", type="password", placeholder="Minimum 6 characters", key="signup_password")
                password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_password_confirm")

                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

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

    # Password reset tab
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: #0d0d10;
                border: 1px solid #00eaff;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 0 20px rgba(0,234,255,0.2);
            ">
                <h3 style="color: #00eaff; text-align: center; margin-bottom: 1.5rem;">Reset Your Password</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("reset_form"):
                email = st.text_input("Email", placeholder="your.email@example.com", key="reset_email")

                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Send Reset Link", use_container_width=True, type="primary")

                if submit:
                    if email:
                        success, message = reset_password(email)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter your email")

    # Footer with back to home link
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
        <p>Don't have time to sign up?</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Continue as Guest", use_container_width=True, type="primary"):
            # Clear the auth page flag so state resets next time
            if 'auth_page_loaded' in st.session_state:
                del st.session_state.auth_page_loaded
            st.session_state.current_page = "Home"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            # Clear the auth page flag so state resets next time
            if 'auth_page_loaded' in st.session_state:
                del st.session_state.auth_page_loaded
            st.session_state.current_page = "Home"
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="VolRegime - Login",
        layout="wide"
    )
    show()
