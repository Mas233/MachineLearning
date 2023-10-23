import lombok.Getter;
import lombok.Setter;
import utils.NodeType;

@Setter
@Getter
public class TreeNode {
    private NodeType type;
    private String value;
    private TreeNode[] children;
}
