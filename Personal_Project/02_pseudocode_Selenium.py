# https://www.youtube.com/watch?v=1b7pXC1-IbE <--- 유튜브 참고 
# https://velog.io/@jungeun-dev/Python-%EC%9B%B9-%ED%81%AC%EB%A1%A4%EB%A7%81Selenium-%EA%B5%AC%EA%B8%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%88%98%EC%A7%91 <-- 


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome() # 크롬드라이버 설치한 경로 작성 필요 
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl") # 구글 이미지 검색 url
elem = driver.find_element_by_name("q") # 구글 검색창 선택
elem.send_keys("pycon") # 검색창에 검색할 내용(name)넣기
elem.send_keys(Keys.RETURN) # 검색할 내용을 넣고 enter를 치는것
time.sleep(5) # n초의 시간동안 대기


# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()
