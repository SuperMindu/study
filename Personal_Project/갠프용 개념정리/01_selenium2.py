from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options 
import time
import os 
import urllib.request


# 이미지들이 createFolder를 지정하여 이미지들이 저장될 폴더를 만들어지게 함
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : Creating directory.' + directory)
        

# 검색어 입력 및 폴더 생성
keyword = 'mark43'
createFolder('d:/study_data/_data/image/PP/' + keyword + '_img_download')

chromedriver = 'c:/study/chromedriver.exe'
driver = webdriver.Chrome(chromedriver)
driver.implicitly_wait(1)

print(keyword, 'searching...')
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
elem = driver.find_element_by_name("q") # 구글 검색창 선택
elem.send_keys(keyword) # 검색창에 검색할 내용(name)넣기
elem.send_keys(Keys.RETURN) # enter 입력

img_folder = './img'
 
if not os.path.isdir(img_folder) : # 없으면 새로 생성하는 조건문 
    os.mkdir(img_folder)
    
    
images = driver.find_elements_by_css_selector("#imgList > div > a > img")
img_url = []
 
for image in images :
    url = image.get_attribute('src')
    img_url.append(url)
    


images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") # 이미지의 클래스 이름을 images 라는 변수에 담아주고 
count = 1
for image in images: # 이미지들 중에 각각 개별 이미지를 하나씩 뽑아서
    image.click() # 그 이미지를 클릭하도록 해줌
    time.sleep(3) # 로딩시간 감안해서 3초 대기하고
    imgUrl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src") # url을 찾아서 
    urllib.request.urlretrieve(imgUrl, 'd:/study_data/_data/image/PP/' + str(count) + ".jpg") # <- 이게 다운받을 폴더명 지정해서 받음
    # urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") # 다운 받도록 만들면 됨 
    count = count + 1


