import subprocess
import time
import pyautogui
import pygetwindow as gw
import json

CHANNELS_JSON = r"D:\animation\LiveStreamTranscription\channels.txt"
CHROME_PATH = r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

# NOTE: Make sure your extension is in slot 1 (first icon), or change the number
EXTENSION_SHORTCUT = "altshift1"  # represents Alt+Shift+1

def open_tabs_in_windows(channels):
    for name, url in channels.items():
        print(f"🟢 Opening: {name} - {url}")
        subprocess.Popen([CHROME_PATH, "--new-window", url])
        time.sleep(3)

def activate_extension_for_tab(tab_title):
    print(f"🎯 Activating extension for tab: {tab_title}")
    win = None
    for _ in range(20):
        wins = [w for w in gw.getWindowsWithTitle(tab_title)]
        if wins:
            win = wins[0]
            break
        time.sleep(0.5)

    if not win:
        print(f"❌ Could not find window for: {tab_title}")
        return

    # Bring window to front
    win.activate()
    time.sleep(1)

    # Open extension popup using keyboard shortcut (Alt+Shift+1)
    pyautogui.hotkey('alt', 'shift', '1')
    time.sleep(1)

    # Navigate inside popup (adjust if needed)
    pyautogui.press('tab')        # focus list
    pyautogui.press('down', presses=3, interval=0.2)  # adjust to reach correct item
    pyautogui.press('enter')      # click Start
    print(f"✅ Extension triggered for {tab_title}")
    time.sleep(1)

def main():
    with open(CHANNELS_JSON, encoding="utf-8") as f:
        channels = json.load(f)

    open_tabs_in_windows(channels)

    for label in channels:
        activate_extension_for_tab(label)

    print("✅ Done.")

if __name__ == "__main__":
    main()
