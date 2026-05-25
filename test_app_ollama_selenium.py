import time
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os

# Fragen und Folgefragen
QUESTIONS = [
    "How do I create a DataFrame from a dictionary?",
    "How can I add a new column to a DataFrame?",
    "How do I filter rows based on a condition?",
    "How do I handle missing values in pandas?",
    "How can I group data by a column?",
    "How do I merge two DataFrames?",
    "How do I sort a DataFrame by multiple columns?",
    "How do I select specific columns from a DataFrame?",
    "How do I calculate the mean of a column?",
    "How do I reset the index of a DataFrame?",
]

FOLLOW_UPS = [
    "Can you show a different example?",
    "What if the column does not exist?",
    "How do I do this with multiple conditions?",
    "How do I fill with the mean value?",
    "How do I get the group sizes?",
    "What if the keys are not unique?",
    "How do I sort descending?",
    "How do I select all except one column?",
    "How do I calculate the median?",
    "How do I keep the old index as a column?",
]

# Screenshots werden hier gespeichert
os.makedirs("screenshots_ollama", exist_ok=True)

# Starte Browser
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1600,1200")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

driver.get("http://localhost:8503")
time.sleep(3)  # Warten bis Streamlit geladen ist



for i in range(10):
    def screenshot_with_timestamp(label):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"screenshots_ollama/q{i+1}_{label}_{ts}.png"
        driver.save_screenshot(fname)
        print(f"Screenshot gespeichert: {fname}")
    # Frage eingeben (warten auf textarea)
    for attempt in range(3):
        try:
            print(f"[Frage {i+1}] Suche textarea für Hauptfrage (Versuch {attempt+1})")
            chat_input = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.TAG_NAME, "textarea"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", chat_input)
            chat_input.clear()
            chat_input.send_keys(QUESTIONS[i])
            chat_input.send_keys(Keys.ENTER)
            print(f"[Frage {i+1}] Hauptfrage gesendet.")
            break
        except Exception as e:
            print(f"Fehler beim Senden der Frage {i+1} (Versuch {attempt+1}): {e}")
            if attempt == 2:
                raise
            time.sleep(3)
    # Warte auf neue Antwort im Chatverlauf (assistant message)
    try:
        WebDriverWait(driver, 30).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, '[data-testid="stChatMessageContent"]')) >= (i+1)*2
        )
    except Exception as e:
        print(f"Timeout beim Warten auf Ollama-Antwort für Frage {i+1}: {e}")
    screenshot_with_timestamp("main")

    # Folgefrage (wieder warten auf neues textarea)
    for attempt in range(3):
        try:
            print(f"[Frage {i+1}] Suche textarea für Folgefrage (Versuch {attempt+1})")
            chat_input = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.TAG_NAME, "textarea"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", chat_input)
            chat_input.clear()
            chat_input.send_keys(FOLLOW_UPS[i])
            chat_input.send_keys(Keys.ENTER)
            print(f"[Frage {i+1}] Folgefrage gesendet.")
            break
        except Exception as e:
            print(f"Fehler beim Senden der Folgefrage {i+1} (Versuch {attempt+1}): {e}")
            if attempt == 2:
                raise
            time.sleep(3)
    # Warte auf neue Antwort im Chatverlauf (assistant message)
    try:
        WebDriverWait(driver, 30).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, '[data-testid="stChatMessageContent"]')) >= (i+1)*2+1
        )
    except Exception as e:
        print(f"Timeout beim Warten auf Ollama-Antwort für Folgefrage {i+1}: {e}")
    screenshot_with_timestamp("followup")

print("Screenshots gespeichert in ./screenshots_ollama/")
driver.quit()
