# created by martinkozle on 31 July 2019
import requests as re
from bs4 import BeautifulSoup as bs
import threading
import io
import time

baseLink = "http://www.makedonski.info"
macedonianCharacters = "абвгдѓежзѕијклљмнњопрстќуфхцчџш"
headers = {
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
	'Accept_Encoding': 'gzip, deflate, br',
	'Accept_Language': 'en-US,en;q=0.9,mk;q=0.8',
}
order = 0   # used to keep order for writing to words in the same order as the website

# multithreading allows for multiple requests to be sent to the website at the same time
class myThread(threading.Thread):
	def __init__(self, function, link, words, position):
		threading.Thread.__init__(self)
		self.function = function
		self.link = link
		self.words = words
		self.position = position

	def run(self):
		values = self.function(self.link)
		global order
		while order != self.position:
			pass
		self.words += values
		order += 1

# returns a list of ranges, for example: [/letter/а/а/абонентски, /leter/а/абонира/авантурист, ...]
def getAllRanges(link):
	global headers
	try:
		html = re.get(link, headers=headers, timeout=100)
	except:
		print("Connection error for " + link + " skipping!")
		return []
	if html.status_code // 100 != 2:
		print(str(html.status_code) + " server error.")
		return []
	soup = bs(html.text, 'lxml')
	# print(soup)
	select = soup.find(id="ranges").find("select")
	options = select.find_all("option")
	ranges = []
	for option in options:
		ranges.append(option.get("value"))
	return ranges

# returns all words in a given range (chrome shows <option> tags, requests shows <a> tags)
def getAllWords(link):
	global headers
	try:
		html = re.get(link, headers=headers, timeout=100)
	except:
		print("Connection error for " + link + " skipping!")
		return []
	if html.status_code // 100 != 2:
		print(str(html.status_code) + " server error.")
		return []
	soup = bs(html.text, 'lxml')
	# print(soup)
	select = soup.find(id="lexems")
	options = select.find_all("a")
	words = []
	for option in options:
		# if it has a gender extension in the word, add it to the list
		if any(rod in option.text for rod in (" м.", " ж.", " ср.")):
			words.append(option.text.replace(" ср.", " c.").replace("  ", ",")[:-1])
	return words


def main():
	file = io.open("data.txt", mode="a", encoding="utf-8")
	global order
	for letter in macedonianCharacters:
		print("Collecting " + letter)
		ranges = getAllRanges(baseLink + "/letter/" + letter)
		words = []
		threads = []
		i = 0
		order = 0
		for range in ranges:
			threads.append(myThread(getAllWords, baseLink + range, words, i))
			threads[-1].start()
			i += 1
		for thread in threads:
			thread.join()
		for word in words:
			print(word)
			file.write(word + "\n")
		# save changes to file after every letter
		file.flush()
	file.close()


if __name__ == "__main__":
	start = time.time()
	main()
	print("[Program finished in " + str(time.time() - start) + "s]")
