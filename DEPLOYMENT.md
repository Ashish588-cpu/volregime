# VolRegime Deployment Guide

Complete guide for deploying VolRegime to Streamlit Cloud.

---

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. Supabase account (optional, for authentication features)
4. Your repository pushed to GitHub

---

## Step 1: Prepare Your Repository

### 1.1 Verify Project Structure

Your project should have this structure:

```
volregime/
├── frontend_v2/
│   ├── app.py                    # Main entry point
│   ├── pages/                    # All page modules
│   ├── components/               # Reusable components
│   ├── utils/                    # Utilities (auth, data_core)
│   ├── styles/                   # Styling modules
│   └── .streamlit/
│       ├── config.toml          # Streamlit configuration
│       └── secrets.toml.example # Example secrets file
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── DEPLOYMENT.md                # This file
```

### 1.2 Verify Requirements File

Your `requirements.txt` should contain:

```
fastapi
uvicorn
pandas
joblib
streamlit
requests
supabase
feedparser
yfinance
plotly
```

### 1.3 Check .gitignore

Ensure `secrets.toml` is in `.gitignore`:

```
.streamlit/secrets.toml
**/secrets.toml
```

**CRITICAL**: Never commit `secrets.toml` to your repository!

---

## Step 2: Push to GitHub

### 2.1 Initialize Git (if not already done)

```bash
cd /path/to/volregime
git init
git add .
git commit -m "Prepare for deployment"
```

### 2.2 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New repository"
3. Name it `volregime`
4. Make it **public** (required for Streamlit Cloud free tier)
5. Click "Create repository"

### 2.3 Push Code

```bash
git remote add origin https://github.com/YOUR_USERNAME/volregime.git
git branch -M main
git push -u origin main
```

**Verify**: Check that `secrets.toml` is NOT visible on GitHub!

---

## Step 3: Set Up Supabase (Optional)

If you want authentication features:

### 3.1 Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign up / Log in
3. Click "New Project"
4. Fill in:
   - **Name**: VolRegime
   - **Database Password**: (save this!)
   - **Region**: Choose closest to your users
5. Click "Create new project"

### 3.2 Get API Credentials

1. In your Supabase project dashboard
2. Click ⚙️ Settings (bottom left)
3. Click "API" in settings menu
4. Copy these values:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **Project API Key** (anon/public key)

**Save these values** - you'll need them for Streamlit Cloud!

---

## Step 4: Deploy to Streamlit Cloud

### 4.1 Sign Up / Log In

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Authorize Streamlit to access your repositories

### 4.2 Create New App

1. Click "New app"
2. Fill in deployment settings:
   - **Repository**: `YOUR_USERNAME/volregime`
   - **Branch**: `main`
   - **Main file path**: `frontend_v2/app.py`
3. Click "Advanced settings"

### 4.3 Configure Secrets (Optional)

If using Supabase authentication:

1. In "Advanced settings", find "Secrets" section
2. Paste this TOML format with your Supabase credentials:

```toml
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "your-anon-key-here"
```

3. Replace the values with your actual Supabase credentials from Step 3.2

### 4.4 Deploy!

1. Click "Deploy!"
2. Wait 2-5 minutes for deployment
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## Step 5: Post-Deployment Configuration

### 5.1 Test Authentication (if enabled)

1. Click "Login" button
2. Try creating an account
3. Check your email for verification link
4. Verify you can sign in

### 5.2 Test Core Features

1. **Home Page**: Verify market data loads
2. **Asset Dashboard**: Search for a ticker (e.g., AAPL)
3. **Portfolio**: Try adding positions
4. **Charts**: Check that Plotly charts render correctly
5. **Navigation**: Verify all pages load without errors

### 5.3 Monitor Logs

1. In Streamlit Cloud dashboard, click "Manage app"
2. View logs to check for any errors
3. Common issues:
   - **Missing dependencies**: Add to `requirements.txt`
   - **Import errors**: Check file paths are correct
   - **API rate limits**: Yahoo Finance has limits, caching helps

---

## Step 6: Custom Domain (Optional)

### 6.1 Streamlit Cloud Custom Domain

1. In your app settings, click "Settings"
2. Find "Custom domain" section
3. Add your domain (requires DNS configuration)
4. Follow Streamlit's DNS setup instructions

### 6.2 Alternative: URL Shortener

Use a service like Bit.ly to create a shorter, branded URL:
- `bit.ly/volregime` → `https://your-app.streamlit.app`

---

## Troubleshooting

### App Won't Start

**Check logs** in Streamlit Cloud for error messages.

Common fixes:
1. Verify `app.py` path is correct: `frontend_v2/app.py`
2. Check all imports resolve correctly
3. Ensure all dependencies are in `requirements.txt`

### Authentication Not Working

1. Verify Supabase secrets are set correctly in Streamlit Cloud
2. Check Supabase project is active (not paused)
3. Test Supabase connection locally first:
   ```bash
   cd frontend_v2
   python test_supabase.py
   ```

### Charts Not Loading

1. Check browser console for JavaScript errors
2. Verify Plotly is in `requirements.txt`
3. Clear browser cache and reload

### Data Not Loading

1. Check Yahoo Finance API is accessible
2. Verify yfinance is in `requirements.txt`
3. Check rate limiting - add caching with `@st.cache_data`

### App is Slow

1. Increase caching TTL values in `@st.cache_data` decorators
2. Optimize data fetching (fetch less data, fetch less frequently)
3. Consider Streamlit Cloud "Professional" tier for better performance

---

## Updating Your App

### Deploy New Changes

```bash
# Make your changes locally
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud will **automatically redeploy** when you push to GitHub!

### Rolling Back

1. In Streamlit Cloud, click "Manage app"
2. Click "Reboot app" to restart with current code
3. Or, revert your Git commit and push:
   ```bash
   git revert HEAD
   git push origin main
   ```

---

## Environment Variables Reference

Set these in Streamlit Cloud "Secrets":

```toml
# Supabase Authentication (optional)
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "eyJxxxxx..."

# Future: Add other API keys here
# OPENAI_API_KEY = "sk-xxxxx"
# NEWS_API_KEY = "xxxxx"
```

---

## Security Best Practices

1. ✅ **Never commit secrets** - use `.gitignore`
2. ✅ **Use HTTPS only** - Streamlit Cloud handles this
3. ✅ **Rotate API keys** regularly in Supabase dashboard
4. ✅ **Enable email verification** in Supabase settings
5. ✅ **Monitor usage** to detect abuse or rate limit issues

---

## Performance Tips

### Caching Strategy

```python
# Short cache for live data (10 seconds)
@st.cache_data(ttl=10)
def get_current_price(ticker):
    pass

# Medium cache for historical data (5 minutes)
@st.cache_data(ttl=300)
def get_historical_data(ticker, period):
    pass

# Long cache for static data (1 hour)
@st.cache_data(ttl=3600)
def get_company_info(ticker):
    pass
```

### Rate Limiting

Yahoo Finance allows **~2000 requests/hour**. Our caching strategy keeps us well below this limit.

---

## Cost Breakdown

### Free Tier (Current Setup)

- **Streamlit Cloud**: Free
- **Supabase**: Free (50,000 monthly active users)
- **Yahoo Finance**: Free (with rate limits)
- **Total**: $0/month

### Paid Tier (If Needed)

- **Streamlit Cloud Pro**: $250/month (better performance, more resources)
- **Supabase Pro**: $25/month (100,000 MAU, better support)
- **Total**: $275/month

Start with free tier and upgrade only if needed!

---

## Support Resources

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **Supabase Docs**: [supabase.com/docs](https://supabase.com/docs)
- **VolRegime Guide**: Navigate to "Guide" page in the app

---

## Next Steps After Deployment

1. **Share your app** with users
2. **Monitor usage** in Streamlit Cloud analytics
3. **Gather feedback** and iterate on features
4. **Add custom domain** for professional branding
5. **Scale up** to paid tiers if traffic grows

---

**Deployment Complete!** 🚀

Your VolRegime app is now live and accessible to users worldwide.

**Last Updated**: December 2024
