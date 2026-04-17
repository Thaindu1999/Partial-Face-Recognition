from supabase import create_client

# 🔥 REPLACE WITH YOUR REAL VALUES
SUPABASE_URL = "https://nspiwffcmseofukduzdi.supabase.co"
SUPABASE_KEY = "sb_publishable_2N6FVHrYZIZk1OHQyYnNRg_NpGhppt8"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)