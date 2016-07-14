
def get_links(text):
    count=False
    links=[]
    newlink=''
    for x in text:
        if x==']':
            count=False
            links.append(newlink)
        if count:
            newlink+=x
        if x=='[':
            count=True
            newlink=''
        return [links[n] for n in range(0,len(links)) if n%2==0]
