import schedule
import time
from crawl_vnexpress import main as crawl_and_store_today_articles
from delete_old import delete_old_articles

def daily_job():
    print("🕒 Starting daily news pipeline...")
    delete_old_articles(days_to_keep=1)
    crawl_and_store_today_articles()
    print("✅ Finished daily update.")

# Schedule at 8AM daily
schedule.every().day.at("08:00").do(daily_job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(20)
