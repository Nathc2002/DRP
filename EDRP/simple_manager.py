from heapq import heappop, heappush
import math


class TaskManager():
    @classmethod
    def assign_task(cls, agent_num, current_tasklist, assigned_tasklist):
        for i in range(agent_num):
            if len(assigned_tasklist[i])==0 and len(current_tasklist)>0 :
                task = current_tasklist.pop(0)
                assigned_tasklist[i] = task
        return current_tasklist, assigned_tasklist
    
    @staticmethod
    def a_star_search(graph, start, goal, positions):
        """
        A* search algorithm implementation for the given graph structure.

        Args:
            graph: NetworkX graph where nodes and edges are defined.
            start: The starting node.
            goal: The goal node.
            positions: Dictionary of node positions for heuristic calculation.

        Returns:
            path (list): The optimal path from start to goal, or None if no path exists.
            path_cost (float): The cost of the path, or float('inf') if no path exists.
        """
        def heuristic(node, goal, positions):
            """Calculate Euclidean distance as the heuristic."""
            if node not in positions or goal not in positions:
                raise ValueError(f"Position for node {node} or goal {goal} not found in positions.")
            x1, y1 = positions[node]
            x2, y2 = positions[goal]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if start not in graph.nodes or goal not in graph.nodes:
            print(f"Error: Start ({start}) or goal ({goal}) not in graph.")
            return None, float('inf')

        # Priority queue for the open set
        open_list = [(0 + heuristic(start, goal, positions), 0, start)]  # (f_score, g_score, node)
        came_from = {}  # To reconstruct the path
        g_score = {node: float('inf') for node in graph.nodes}  # Cost from start to each node
        g_score[start] = 0

        while open_list:
            # Get the node with the smallest f_score
            _, current_g, current_node = heappop(open_list)

            # If the goal is reached, reconstruct the path
            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                path.reverse()
                return path, current_g

            # Process neighbors of the current node
            for neighbor in graph.neighbors(current_node):
                edge_weight = graph[current_node][neighbor].get('weight', 1)  # Default weight is 1
                tentative_g_score = current_g + edge_weight

                # If a better path to the neighbor is found
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal, positions)
                    heappush(open_list, (f_score, tentative_g_score, neighbor))

        # If the open list is empty and goal not reached, no path exists
        print(f"No path found from {start} to {goal}.")
        return None, float('inf')


    @staticmethod
    def get_smallest_cost(couples):
        """
        Find the tuple (agent, cost) with the smallest cost.

    @param couples: List of tuples where each tuple is (agent, cost).
    @return: The tuple with the smallest cost.
        """
        return min(couples, key=lambda x: x[1])
    

    @staticmethod
    def assign_task_v2(agent_num, current_tasklist, assigned_tasklist, graph, obs, positions, start_ori_array):
        candidates = []
        costs = []

        # Find agents with no assigned tasks
        for i in range(agent_num):
            if len(assigned_tasklist[i]) == 0 and len(current_tasklist) > 0:
                candidates.append(i)

        if not candidates:
            print("No candidates available for task assignment.")
            return current_tasklist, assigned_tasklist

        # For each candidate agent, compute the cost to reach the first task
        for j in candidates:
            try:
                # Instead of extracting the start node from obs, use start_ori_array
                start_node = start_ori_array[j]

                if len(current_tasklist) == 0:
                    print("No tasks available in current_tasklist.")
                    break

                task_start_node = int(current_tasklist[0][0])
                path, cost = TaskManager.a_star_search(graph, start_node, task_start_node, positions)
                
                if path is not None:
                    costs.append((j, cost))
                else:
                    print(f"No valid path found for agent {j} from {start_node} to {task_start_node}.")
            
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error processing agent {j}: {e}")
                continue

        print(f"Costs: {costs}")

        # Assign the task to the agent with the smallest cost
        if costs:
            selected_agent = TaskManager.get_smallest_cost(costs)[0]
            task = current_tasklist.pop(0)
            assigned_tasklist[selected_agent] = task
            print(f"Task {task} has been assigned to agent {selected_agent}")
        else:
            print("No valid agents found to assign tasks.")

        return current_tasklist, assigned_tasklist
