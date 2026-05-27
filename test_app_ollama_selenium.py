import time
from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException

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

LLM_TIMEOUT = 120   # seconds to wait for llama3.1:8b response

os.makedirs("screenshots_ollama", exist_ok=True)


def make_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--blink-settings=imagesEnabled=false")
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)


driver = make_driver()
driver.get("http://localhost:8503")

print("Warte auf Chat-Input (max 60s)...")
try:
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stChatInput"] textarea'))
    )
    print("Chat-Input gefunden.")
except Exception:
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "textarea"))
    )


def screenshot(label, i):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"screenshots_ollama/q{i+1}_{label}_{ts}.png"
    try:
        driver.save_screenshot(fname)
        print(f"Screenshot gespeichert: {fname}")
    except (InvalidSessionIdException, WebDriverException) as e:
        print(f"Screenshot fehlgeschlagen (Browser weg?): {e}")


def wait_for_streaming_complete(timeout=180):
    """Wait until the Streamlit Stop button disappears (streaming is done)."""
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-testid="stStopButton"]'))
        )
    except Exception:
        pass  # Stop button may not have appeared yet; that's fine
    try:
        WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-testid="stStopButton"]'))
        )
    except Exception:
        pass  # Timeout is acceptable — screenshot will capture current state


def send_message(text, i, label):
    for attempt in range(3):
        try:
            print(f"[Frage {i+1}] Suche textarea für {label} (Versuch {attempt+1})")
            chat_input = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="stChatInput"] textarea'))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", chat_input)
            chat_input.clear()
            chat_input.send_keys(text)
            chat_input.send_keys(Keys.ENTER)
            print(f"[Frage {i+1}] {label} gesendet.")
            return True
        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"Browser-Session verloren bei {label} {i+1}: {e}")
            return False
        except Exception as e:
            print(f"Fehler beim Senden ({label} {i+1}, Versuch {attempt+1}): {e}")
            if attempt == 2:
                return False
            time.sleep(3)
    return False


for i in range(10):
    try:
        # Hauptfrage
        if not send_message(QUESTIONS[i], i, "Hauptfrage"):
            print(f"[Frage {i+1}] übersprungen (Browser-Fehler)")
            continue

        expected_main = (i + 1) * 2
        try:
            WebDriverWait(driver, LLM_TIMEOUT).until(
                lambda d, n=expected_main: len(
                    d.find_elements(By.CSS_SELECTOR, '[data-testid="stChatMessageContent"]')
                ) >= n
            )
        except Exception as e:
            print(f"Timeout Hauptfrage {i+1}: {e}")
        screenshot("main", i)
        wait_for_streaming_complete()  # ensure memory is updated before follow-up

        # Folgefrage
        if not send_message(FOLLOW_UPS[i], i, "Folgefrage"):
            print(f"[Folgefrage {i+1}] übersprungen (Browser-Fehler)")
            continue

        expected_followup = (i + 1) * 2 + 1
        try:
            WebDriverWait(driver, LLM_TIMEOUT).until(
                lambda d, n=expected_followup: len(
                    d.find_elements(By.CSS_SELECTOR, '[data-testid="stChatMessageContent"]')
                ) >= n
            )
        except Exception as e:
            print(f"Timeout Folgefrage {i+1}: {e}")
        screenshot("followup", i)

    except (InvalidSessionIdException, WebDriverException) as e:
        print(f"[Frage {i+1}] Browser-Session verloren, überspringe Rest: {e}")
        break
    except Exception as e:
        print(f"[Frage {i+1}] Unerwarteter Fehler: {e}")
        continue

print("Screenshots gespeichert in ./screenshots_ollama/")
try:
    driver.quit()
except Exception:
    pass
