import bipartite_graph
import bipartite_graph_demo
import variability_graph

def main_menu():
    print("")
    print("Select which script to run:")
    print("1 - Run Semantic Analysis - displays vertices representing all images and keywords, and edges representing the relationship between them.")
    print("2 - Run Semantic Analysis Image Demo - displays a small amount of random images and their keywords, and edges representing the relationship between them.")
    print("3 - Run Variability Graph - displays a graph of the variability of the similarity scores of the images.")
    print("0 - Exit")
    choice = input("Enter choice: ")
    return choice

def main():
    while True:
        choice = main_menu()
        
        if choice == "1":
            bipartite_graph.run_semantic_analysis() 
        elif choice == "2":
            bipartite_graph_demo.run_semantic_analysis_demo()  
        elif choice == "3":
            variability_graph.run_variability_graph()
        elif choice == "0":
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
