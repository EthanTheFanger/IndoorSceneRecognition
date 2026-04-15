import os
import shutil

# Root directory containing all the room folders
BASE_DIR = "Images"

# Category mapping
indoor_taxonomy = {
    "public_spaces": [
        "airport_inside", "trainstation", "subway", "inside_subway",
        "inside_bus", "waitingroom", "auditorium", "church_inside",
        "cloister", "library", "museum", "locker_room",
        "elevator", "prisoncell", "poolinside",
    ],
    "leisure": [
        "casino", "gameroom", "gym", "bar", "restaurant",
        "fastfood_restaurant", "movietheater", "concert_hall",
        "hairsalon", "buffet", "bowling",
    ],
    "store": [
        "mall", "bookstore", "clothingstore", "videostore", "grocerystore",
        "jewelleryshop", "toystore", "shoeshop", "florist",
        "laundromat", "deli", "bakery",
    ],
    "working_spaces": [
        "office", "computerroom", "tv_studio", "studiomusic", "artstudio",
        "laboratorywet", "warehouse", "meeting_room", "classroom",
        "kindergarden", "hospitalroom", "operating_room", "dentaloffice",
        "restaurant_kitchen", "greenhouse",
    ],
    "home": [
        "bedroom", "bathroom", "livingroom", "dining_room", "kitchen",
        "closet", "pantry", "nursery", "children_room", "corridor",
        "stairscase", "garage", "lobby", "winecellar",
    ],
}

# Flat reverse lookup: room → category
room_to_category = {
    room: category
    for category, rooms in indoor_taxonomy.items()
    for room in rooms
}

# Flat reverse lookup: room → category
room_to_category = {
    room: category
    for category, rooms in indoor_taxonomy.items()
    for room in rooms
}


def organize_dataset(base_dir: str):
    # Step 1: Create the 5 mega-category folders
    for category in indoor_taxonomy:
        category_path = os.path.join(base_dir, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"Created folder: {category}/")

    print()

    # Step 2: Move each room folder into its category folder
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)

        # Skip if not a directory or is one of our 5 category folders
        if not os.path.isdir(item_path) or item in indoor_taxonomy:
            continue

        category = room_to_category.get(item)

        if category:
            dest = os.path.join(base_dir, category, item)
            shutil.move(item_path, dest)
            print(f"Moved: {item:<25} → {category}/")
        else:
            print(f"Skipped (uncategorized): {item}")


organize_dataset(BASE_DIR)