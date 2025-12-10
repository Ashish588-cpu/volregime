# VolRegime Deployment Checklist

Quick checklist to ensure everything is ready for deployment.

---

## Pre-Deployment Checklist

### ✅ Code Preparation

- [ ] All recent changes committed to Git
- [ ] `requirements.txt` contains all dependencies
- [ ] `.gitignore` includes `secrets.toml`
- [ ] No hardcoded API keys or secrets in code
- [ ] All imports working locally
- [ ] App runs without errors locally

### ✅ File Structure

- [ ] Main file at `frontend_v2/app.py`
- [ ] All pages in `frontend_v2/pages/`
- [ ] All utilities in `frontend_v2/utils/`
- [ ] Config at `frontend_v2/.streamlit/config.toml`
- [ ] Secrets example at `frontend_v2/.streamlit/secrets.toml.example`

### ✅ GitHub

- [ ] Repository created on GitHub
- [ ] Repository is **public** (required for free tier)
- [ ] Code pushed to `main` branch
- [ ] Verify `secrets.toml` is NOT in GitHub (check online)

### ✅ Supabase (Optional)

- [ ] Supabase project created
- [ ] Project URL copied
- [ ] API key (anon) copied
- [ ] Authentication enabled in Supabase settings

---

## Deployment Steps

### 1. Streamlit Cloud Setup

- [ ] Sign up at [share.streamlit.io](https://share.streamlit.io)
- [ ] Connect GitHub account
- [ ] Authorize repository access

### 2. Create App

- [ ] Click "New app"
- [ ] Select repository: `YOUR_USERNAME/volregime`
- [ ] Set branch: `main`
- [ ] Set main file: `frontend_v2/app.py`

### 3. Configure Secrets (Optional)

- [ ] Click "Advanced settings"
- [ ] Add Supabase secrets:
  ```toml
  SUPABASE_URL = "https://xxxxx.supabase.co"
  SUPABASE_KEY = "your-anon-key"
  ```

### 4. Deploy

- [ ] Click "Deploy!"
- [ ] Wait for deployment (2-5 minutes)
- [ ] Check for errors in logs

---

## Post-Deployment Testing

### Core Features

- [ ] Home page loads with market data
- [ ] Scrolling ticker bar works
- [ ] Navigation bar works (purple with bold text)
- [ ] Asset Dashboard loads
- [ ] Can search for tickers (try: AAPL, MSFT, SPY)
- [ ] Charts render correctly
- [ ] Portfolio page loads
- [ ] Guide page accessible

### Authentication (if enabled)

- [ ] Login button appears
- [ ] Can click Login and see Auth page
- [ ] Can create new account
- [ ] Receive verification email
- [ ] Can sign in with credentials
- [ ] User email displays when logged in
- [ ] Sign Out works

### Performance

- [ ] Pages load within 3 seconds
- [ ] Data updates (check timestamp)
- [ ] No console errors
- [ ] Charts are interactive
- [ ] Mobile responsive (check on phone)

---

## Troubleshooting

If something doesn't work:

1. **Check Streamlit Cloud logs**
   - Click "Manage app" → View logs
   - Look for error messages

2. **Common Issues**:
   - Import errors → Check file paths
   - Missing dependencies → Update `requirements.txt`
   - Auth not working → Check Supabase secrets
   - Slow loading → Add more caching

3. **Test Locally First**:
   ```bash
   cd frontend_v2
   streamlit run app.py
   ```

---

## Quick Commands

### Push Updates

```bash
git add .
git commit -m "Your message"
git push origin main
```

### Revert Changes

```bash
git revert HEAD
git push origin main
```

### Check Git Status

```bash
git status
git log --oneline -5
```

---

## Final Verification

Before marking complete:

- [ ] App URL works: `https://your-app.streamlit.app`
- [ ] All pages accessible
- [ ] No errors in logs
- [ ] Authentication works (if enabled)
- [ ] Shared URL with at least one person who confirmed it works

---

**Deployment Date**: _______________

**App URL**: _______________

**Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Complete

---

## Next Steps

After successful deployment:

1. Share app URL with users
2. Monitor usage in Streamlit Cloud
3. Gather feedback
4. Plan next features
5. Consider custom domain

---

**Need Help?**
- Read: [DEPLOYMENT.md](DEPLOYMENT.md)
- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Streamlit Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
