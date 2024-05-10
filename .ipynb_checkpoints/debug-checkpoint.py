def d_print(func, str):
    with open('debug.txt', 'a') as f:
        f.write(f"(In {func}) {str}\n");
        
def d_start():
    with open('debug.txt', 'a') as f:
        f.write(f"\nProgram launching\n");