def preprocess_chromosone():
    with open('chr178.txt', 'r') as content_file:
        text = content_file.read()[:1000000]
    with open('chrBAD.txt','w',encoding='utf8',newline='') as content_file:
        content_file.write(text)

preprocess_chromosone()