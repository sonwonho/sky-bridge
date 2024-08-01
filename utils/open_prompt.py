def open_prompt(path: str) -> str:
    with open(path, "r") as text:
        return "".join(text.readlines())


if __name__ == "__main__":
    text = open_prompt("prompt/gooroom_user.txt")
    year = 2027
    subject = "국어"
    grade = 4
    university = "서울대"
    print(text.format(year=year, subject=subject, grade=grade, university=university))
