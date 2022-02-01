def pepify_word(word):


    for i, letter in enumerate(word):
        if letter.isupper():
            if i == 0 or not word[i - 1].isalnum():
                word = word.replace(letter, letter.lower())
            else:
                word = word.replace(letter, f"_{letter.lower()}")

    return word


def pepify_text(fh, out):

    new = []
    for line in fh:
        new.append(" ".join((pepify_word(word) for word in line.split(' '))))


    with open(out, "w") as fo:
        fo.writelines(new)

with open("Homogeneity_report.py", "r") as fh:
    pepify_text(fh, "homogeneity_report.py")
