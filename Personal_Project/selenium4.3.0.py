from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import Request, urlopen 

# selenium 최신버전으로 문법이 바꼈다. 바뀐걸로 적용해줌.
# Chrome 드라이버 자동으로 잡아주는게 추가됨(Service, ChromeDriverManager)
# Xpath 위치 잘 봐야함

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def crawling_img(name):
    print('검색중...')
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install())) #드라이버 위치 자동으로 찾아줌
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem = driver.find_element(By.NAME,"q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)
    
    print('스크롤 내리는 중.. 좀만 ㄱㄷ')
    SCROLL_PAUSE_TIME = 1
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")  # 브라우저의 높이를 자바스크립트로 찾음
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내림
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_elements(By.CSS_SELECTOR,".mye4qd").click()
            except:
                break
        last_height = new_height

    imgs = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
    dir = "D:/PP"+ "/" + name
    # 끝까지 다내려서 개느림
    print('이미지 다운중.. 좀만 ㄱㄷ')
    
    createDirectory(dir) # 폴더 생성해준다
    count = 1
    for img in imgs:
        try:
            img.click()
            time.sleep(2.5)
            imgUrl = driver.find_element(By.XPATH,'//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
            path = "D:/PP/" + name + "/"
            urllib.request.urlretrieve(imgUrl, path + f"{name}" + str(count) + ".jpg")
            count = count + 1
            img_number = 300
            if count >= img_number: # 이미지 장수 선택 
                break
        except:
            print("path 다시 확인 해라!!!") # 경로못찾으면 패~쓰~~~~~
    driver.close()
images = ["전신샷"]

for image in images:
    crawling_img(image)
    
