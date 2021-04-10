def count():
    with open('lastfm-kg15k/kg.txt') as f:
        count = len(f.readlines())
    print(count)


def cut():
    import os
    from Recommender_System.data.data_loader import book_crossing
    data = book_crossing()
    item_set = {d[1] for d in data}
    delete_item_list = []
    lines = []
    directory = 'bx-kg150k'

    with open(os.path.join(directory, 'item_id2entity_id_old.txt'), 'r') as reader:
        for line in reader.readlines():
            item_id = line.strip().split('\t')[0]  # str
            if item_id in item_set:
                lines.append(line)
            else:
                delete_item_list.append(item_id)

    with open(os.path.join(directory, 'item_id2entity_id.txt'), 'w', encoding='utf-8') as writer:
        writer.writelines(lines)

    print(len(delete_item_list))
    print(delete_item_list)


if __name__ == '__main__':
    cut()
