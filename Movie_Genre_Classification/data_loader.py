import re

def load_data(file_path, has_genre=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if has_genre and len(parts) == 4:
                id_, title, genre, plot = parts
                data.append((title, genre.split(','), plot))
            elif not has_genre and len(parts) == 3:
                id_, title, plot = parts
                data.append((title, plot))
    return data
