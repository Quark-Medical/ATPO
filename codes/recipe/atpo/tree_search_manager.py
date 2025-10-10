import uuid
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage

class TreeSearchManager:
    """Manages the state of the breadth-first tree search."""

    def __init__(self, initial_messages: List[BaseMessage], M: int):
        """
        Initializes the manager.
        Args:
            initial_messages (List[BaseMessage]): The starting messages (system + first human prompt).
            M (int): The target number of trajectories to generate.
        """
        self.M = M
        # Each trajectory is a list of BaseMessages
        self.active_trajectories: List[List[BaseMessage]] = [initial_messages]
        self.completed_trajectories: List[List[BaseMessage]] = []

    def is_done(self) -> bool:
        """Checks if the search is complete."""
        return len(self.completed_trajectories) >= self.M or not self.active_trajectories

    def get_trajectories_to_expand(self) -> List[List[BaseMessage]]:
        """Returns the current set of active trajectories (leaves of the tree)."""
        return self.active_trajectories

    def expand_and_replace(self, expansions: List[Tuple[List[BaseMessage], List[BaseMessage]]]):
        """
        Replaces the current active trajectories with their new expansions.
        This is the core of the BFS level transition.
        
        Args:
            expansions (List[Tuple[List[BaseMessage], List[BaseMessage]]]): A list where each
                tuple contains (original_trajectory, list_of_new_ai_or_user_messages]).
        """
        new_active_trajectories = []
        for original_traj, new_nodes in expansions:
            if not isinstance(new_nodes, list):
                new_nodes = [new_nodes] # Handle single user message case
            for new_node in new_nodes:
                new_active_trajectories.append(original_traj + [new_node])
        self.active_trajectories = new_active_trajectories

    def prune_and_select(self, trajectories_to_keep: List[List[BaseMessage]]):
        """Prunes the tree by replacing active trajectories with a smaller subset."""
        self.active_trajectories = trajectories_to_keep

    def complete_trajectories(self, trajectories_to_complete: List[List[BaseMessage]]):
        """Moves trajectories from the active list to the completed list."""
        for traj in trajectories_to_complete:
            if len(self.completed_trajectories) < self.M:
                if traj not in self.completed_trajectories:
                     self.completed_trajectories.append(traj)

        self.active_trajectories = [t for t in self.active_trajectories if t not in trajectories_to_complete]

    def get_final_results(self) -> List[List[BaseMessage]]:
        """Returns the collected completed trajectories."""
        return self.completed_trajectories

@dataclass
class TreeNode:
    """Represents a node in the search tree."""
    uid: str = field(init=False, default_factory=lambda: str(uuid.uuid4())) 
    messages: List[BaseMessage]                    
    parent: Optional['TreeNode'] = None            
    children: List['TreeNode'] = field(default_factory=list)
    is_terminal: bool = False
    explore_num: int = 0                            
    level: int = 0                              
    q_value_variance: float = None                  
    
    action_reward: Optional[float] = None          
    critic_state_value: Optional[float] = None     
    mdp_state_value: Optional[float] = None        
    q_value: Optional[float] = None                

    advantage: Optional[float] = None             
    returns: Optional[float] = None              
    
    assistant_message: Optional[BaseMessage] = None 
    verifier_response: Optional[str] = None        

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.uid == other.uid