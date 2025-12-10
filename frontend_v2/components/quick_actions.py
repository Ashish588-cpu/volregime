"""
Quick Actions Component - Floating action button with keyboard shortcuts
Provides quick access to common actions across all pages
"""

import streamlit as st


def render_quick_actions():
    """
    Render the floating quick actions button and menu.
    Call this at the end of app.py to display on all pages.
    """
    
    # Initialize session state for quick actions
    if'quick_action_modal' not in st.session_state:
        st.session_state.quick_action_modal = None
    if'quick_search_ticker' not in st.session_state:
        st.session_state.quick_search_ticker =""
    
    # Inject the floating button and menu HTML/CSS/JS
    quick_actions_html ="""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    #quick-actions-btn {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #4f8df9, #4752fa);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(79, 141, 249, 0.5), 0 0 40px rgba(79, 141, 249, 0.2);
        z-index: 10000;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        outline: none;
    }
    
    #quick-actions-btn:hover {
        transform: scale(1.1) rotate(15deg);
        box-shadow: 0 6px 30px rgba(79, 141, 249, 0.6), 0 0 60px rgba(79, 141, 249, 0.3);
    }
    
    #quick-actions-btn:active {
        transform: scale(0.95);
    }
    
    #quick-actions-btn span {
        font-size: 1.5rem;
        color: white;
        user-select: none;
    }
    
    #quick-actions-menu {
        position: fixed;
        bottom: 92px;
        right: 24px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 0;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(51, 65, 85, 0.5);
        z-index: 10001;
        min-width: 280px;
        opacity: 0;
        visibility: hidden;
        transform: translateY(10px) scale(0.95);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    #quick-actions-menu.show {
        opacity: 1;
        visibility: visible;
        transform: translateY(0) scale(1);
    }
    
    .qa-header {
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, #4f8df9, #4752fa);
        color: white;
    }
    
    .qa-header h4 {
        margin: 0;
        font-family:'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .qa-header p {
        margin: 0.25rem 0 0 0;
        font-size: 0.75rem;
        opacity: 0.9;
    }
    
    .qa-menu-items {
        padding: 0.5rem 0;
    }
    
    .qa-menu-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        cursor: pointer;
        transition: background 0.15s;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
        font-family:'Poppins', sans-serif;
    }
    
    .qa-menu-item:hover {
        background: #334155;
    }

    .qa-menu-item-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .qa-menu-item-icon {
        font-size: 1.1rem;
    }

    .qa-menu-item-label {
        font-size: 0.9rem;
        color: #f8fafc;
        font-weight: 500;
    }

    .qa-menu-item-shortcut {
        font-size: 0.7rem;
        color: #94a3b8;
        background: #0f172a;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: monospace;
    }

    .qa-divider {
        height: 1px;
        background: #334155;
        margin: 0.25rem 0;
    }
    
    #qa-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 9999;
        display: none;
    }
    
    #qa-overlay.show {
        display: block;
    }
    
    /* Keyboard shortcut toast */
    #shortcut-toast {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%) translateY(20px);
        background: #1e293b;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-family:'Poppins', sans-serif;
        font-size: 0.9rem;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s;
        z-index: 10002;
    }
    
    #shortcut-toast.show {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
    }
    </style>

    <!-- Overlay to close menu when clicking outside -->
    <div id="qa-overlay" onclick="closeQuickActions()"></div>

    <!-- Quick Actions Menu -->
    <div id="quick-actions-menu">
        <div class="qa-header">
            <h4> Quick Actions</h4>
            <p>Press shortcuts or click below</p>
        </div>
        <div class="qa-menu-items">
            <button class="qa-menu-item" onclick="handleAction('search')">
                <span class="qa-menu-item-left">
                    <span class="qa-menu-item-icon"></span>
                    <span class="qa-menu-item-label">Search Ticker</span>
                </span>
                <span class="qa-menu-item-shortcut">Ctrl+K</span>
            </button>
            <button class="qa-menu-item" onclick="handleAction('watchlist')">
                <span class="qa-menu-item-left">
                    <span class="qa-menu-item-icon">⭐</span>
                    <span class="qa-menu-item-label">View Watchlist</span>
                </span>
                <span class="qa-menu-item-shortcut">Ctrl+W</span>
            </button>
            <button class="qa-menu-item" onclick="handleAction('compare')">
                <span class="qa-menu-item-left">
                    <span class="qa-menu-item-icon"></span>
                    <span class="qa-menu-item-label">Compare Assets</span>
                </span>
                <span class="qa-menu-item-shortcut">Ctrl+C</span>
            </button>
            <button class="qa-menu-item" onclick="handleAction('news')">
                <span class="qa-menu-item-left">
                    <span class="qa-menu-item-icon"></span>
                    <span class="qa-menu-item-label">Latest News</span>
                </span>
                <span class="qa-menu-item-shortcut">Ctrl+N</span>
            </button>
            <div class="qa-divider"></div>
            <button class="qa-menu-item" onclick="handleAction('help')">
                <span class="qa-menu-item-left">
                    <span class="qa-menu-item-icon"></span>
                    <span class="qa-menu-item-label">Help & Shortcuts</span>
                </span>
                <span class="qa-menu-item-shortcut">Ctrl+H</span>
            </button>
        </div>
    </div>

    <!-- Floating Action Button -->
    <div id="quick-actions-btn" onclick="toggleQuickActions()" title="Quick Actions (Ctrl+/)">
        <span></span>
    </div>

    <!-- Toast notification -->
    <div id="shortcut-toast"></div>

    <script>
    let menuOpen = false;

    function toggleQuickActions() {
        menuOpen = !menuOpen;
        const menu = document.getElementById('quick-actions-menu');
        const overlay = document.getElementById('qa-overlay');
        const btn = document.getElementById('quick-actions-btn');

        if (menuOpen) {
            menu.classList.add('show');
            overlay.classList.add('show');
            btn.innerHTML ='<span></span>';
        } else {
            menu.classList.remove('show');
            overlay.classList.remove('show');
            btn.innerHTML ='<span></span>';
        }
    }

    function closeQuickActions() {
        if (menuOpen) {
            toggleQuickActions();
        }
    }

    function showToast(message) {
        const toast = document.getElementById('shortcut-toast');
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
    }

    function handleAction(action) {
        closeQuickActions();

        // Use Streamlit's query params to trigger actions
        const baseUrl = window.location.origin + window.location.pathname;

        switch(action) {
            case'search':
                showToast(' Opening ticker search...');
                // Set session state via URL or trigger Streamlit action
                window.parent.postMessage({type:'streamlit:setComponentValue', value: {action:'search'}},'*');
                break;
            case'watchlist':
                showToast('⭐ Opening watchlist...');
                window.parent.postMessage({type:'streamlit:setComponentValue', value: {action:'watchlist'}},'*');
                break;
            case'compare':
                showToast(' Opening asset comparison...');
                window.parent.postMessage({type:'streamlit:setComponentValue', value: {action:'compare'}},'*');
                break;
            case'news':
                showToast(' Opening news...');
                window.parent.postMessage({type:'streamlit:setComponentValue', value: {action:'news'}},'*');
                break;
            case'help':
                showToast(' Opening help...');
                window.parent.postMessage({type:'streamlit:setComponentValue', value: {action:'help'}},'*');
                break;
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+/ to toggle menu
        if (e.ctrlKey && e.key ==='/') {
            e.preventDefault();
            toggleQuickActions();
            return;
        }

        // Only handle shortcuts when Ctrl is pressed
        if (!e.ctrlKey) return;

        switch(e.key.toLowerCase()) {
            case'k':
                e.preventDefault();
                handleAction('search');
                break;
            case'w':
                e.preventDefault();
                handleAction('watchlist');
                break;
            case'c':
                e.preventDefault();
                handleAction('compare');
                break;
            case'n':
                e.preventDefault();
                handleAction('news');
                break;
            case'h':
                e.preventDefault();
                handleAction('help');
                break;
        }
    });

    // Close menu on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key ==='Escape' && menuOpen) {
            closeQuickActions();
        }
    });
    </script>
    """

    # Render the HTML component using st.markdown for better integration
    st.markdown(quick_actions_html, unsafe_allow_html=True)


def render_help_modal():
    """Render the help/shortcuts modal when triggered"""

    if st.session_state.get('show_help_modal', False):
        with st.container():
            st.markdown("""
            <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9998;"></div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="background: white; border-radius: 16px; padding: 2rem; position: relative; z-index: 9999; margin-top: 100px;">
                    <h2 style="margin: 0 0 1rem 0; font-family:'Poppins', sans-serif;"> Help & Keyboard Shortcuts</h2>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### ⌨️ Keyboard Shortcuts")

                shortcuts_data = {
                    "Ctrl + /":"Toggle Quick Actions menu",
                    "Ctrl + K":"Search for a ticker",
                    "Ctrl + W":"View your watchlist",
                    "Ctrl + C":"Compare assets",
                    "Ctrl + N":"Open latest news",
                    "Ctrl + H":"Open this help dialog",
                    "Escape":"Close any open menu/modal"
                }

                for shortcut, description in shortcuts_data.items():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;">
                        <span style="color: #64748b;">{description}</span>
                        <code style="background: #f1f5f9; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem;">{shortcut}</code>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### Quick Tips")
                st.markdown("""
                - Use the **⭐ Star button** on Asset Dashboard to add tickers to your watchlist
                - **Bookmark articles** in the News section for later reading
                - Track your **learning progress** in the Articles / Learn section
                - View **regime performance** of your portfolio in Portfolio Tools
                """)

                if st.button("Close", key="close_help_modal"):
                    st.session_state.show_help_modal = False
                    st.rerun()


def render_search_modal():
    """Render quick search modal"""

    if st.session_state.get('show_search_modal', False):
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 20px 50px rgba(0,0,0,0.2); margin-top: 100px;">
                    <h3 style="margin: 0 0 1rem 0; font-family:'Poppins', sans-serif;"> Quick Search</h3>
                </div>
                """, unsafe_allow_html=True)

                search_ticker = st.text_input(
                    "Enter ticker symbol",
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    key="quick_search_input"
                ).upper()

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Search", use_container_width=True, type="primary"):
                        if search_ticker:
                            st.session_state.selected_ticker = search_ticker
                            st.session_state.show_search_modal = False
                            # Would need to navigate to Asset Dashboard
                            st.rerun()
                with col_b:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.show_search_modal = False
                        st.rerun()


def render_watchlist_sidebar():
    """Render watchlist in a sidebar overlay"""

    if st.session_state.get('show_watchlist_overlay', False):
        with st.sidebar:
            st.markdown("### ⭐ Quick Watchlist")
            st.markdown("---")

            watchlist = st.session_state.get('watchlist', [])

            if watchlist:
                for ticker in watchlist[:10]: # Show max 10
                    if st.button(f" {ticker}", key=f"qw_{ticker}", use_container_width=True):
                        st.session_state.selected_ticker = ticker
                        st.session_state.show_watchlist_overlay = False
                        st.rerun()
            else:
                st.info("No tickers in watchlist. Add some from the Asset Dashboard!")

            st.markdown("---")
            if st.button("Close", key="close_watchlist_overlay"):
                st.session_state.show_watchlist_overlay = False
                st.rerun()

