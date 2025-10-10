import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union
import colorsys
import torch
from model2vec.model import StaticModel
from llm2vec import LLM2Vec


class TokenEmbeddingVisualizer:
    """
    Unified visualizer for token embeddings from both static and transformer models.
    Automatically detects model type and visualizes embeddings as connected arrows 
    with optional attention edges.
    """
    
    def __init__(self, model=None, device: str = None):
        """
        Initialize the visualizer.
        
        Args:
            model: Embedding model (static or LLM2Vec)
            device: Device for torch operations ('cuda' or 'cpu')
        """
        self.model = model
        self.model_type = self._detect_model_type()
        self.pca = PCA(n_components=2)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def _detect_model_type(self) -> str:
        """
        Automatically detect model type using isinstance checks.
        
        Returns:
            'static' or 'llm2vec'
        """
        if isinstance(self.model, StaticModel):
            return 'static'
        
        if isinstance(self.model, LLM2Vec):
                return 'llm2vec'
        
        raise ValueError(f"Unsupported model type: {type(self.model)}. "
                     f"Expected StaticModel or LLM2Vec.")


    def _calculate_l2_norm(self, vector: np.ndarray) -> float:
        """Calculate L2 norm of a vector."""
        return np.linalg.norm(vector)
    
    def _generate_colors(self, n_colors: int) -> List[str]:
        """Generate distinct colors for different strings."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
        return colors
    
    def _extract_static_embeddings(self, strings: List[str]) -> Tuple[List[List[Dict]], List[str]]:
        """
        Extract embeddings from static model.
        
        Returns:
            tokens_data: List of token data for each string
            all_tokens: Flattened list of all tokens
        """
        embeddings_list = self.model.encode_as_sequence(strings)
        all_tokens_data = []
        all_tokens = []
        
        for string_idx, (text, embeddings_array) in enumerate(zip(strings, embeddings_list)):
            tokens = self.model.tokenizer.encode(text).tokens
            token_data = []
            
            for token_idx, (token, embedding) in enumerate(zip(tokens, embeddings_array)):
                token_data.append({
                    'token': token,
                    'embedding': embedding,
                    'l2_norm': self._calculate_l2_norm(embedding),
                    'string_idx': string_idx,
                    'token_idx': token_idx
                })
                all_tokens.append(token)
            
            all_tokens_data.append(token_data)
        
        return all_tokens_data, all_tokens
    
    def _extract_llm2vec_embeddings(self, strings: List[str], 
                                   exclude_special_tokens: bool = True) -> Tuple[List[List[Dict]], List[str], Optional[List[np.ndarray]]]:
        """
        Extract embeddings and attention from LLM2Vec model.
        
        Returns:
            tokens_data: List of token data for each string
            all_tokens: Flattened list of all tokens
            attention_matrices: List of attention matrices (one per string)
        """
        if self.model_type != 'llm2vec':
            raise ValueError("This method requires model_type='llm2vec'")
        
        all_tokens_data = []
        all_tokens = []
        attention_matrices = []
        
        for string_idx, text in enumerate(strings):
            # Tokenize
            tokenizer = self.model.tokenizer
            inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get tokens
            tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                )
            
            # Extract embeddings and attention
            last_hidden_state = outputs.hidden_states[-1][0]  # Remove batch dim
            last_layer_attention = outputs.attentions[-1][0]  # Remove batch dim
            
            # Average attention across heads: [num_heads, seq_len, seq_len] -> [seq_len, seq_len]
            avg_attention = last_layer_attention.mean(dim=0)
            
            # Convert to numpy
            embeddings = last_hidden_state.float().cpu().numpy()
            attention = avg_attention.float().cpu().numpy()
            
            # Filter special tokens if requested
            if exclude_special_tokens:
                special_tokens = set(tokenizer.all_special_tokens)
                keep_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
                
                if keep_indices:
                    tokens = [tokens[i] for i in keep_indices]
                    embeddings = embeddings[keep_indices]
                    # Filter attention matrix (both dimensions)
                    attention = attention[np.ix_(keep_indices, keep_indices)]
            
            # Build token data
            token_data = []
            for token_idx, (token, embedding) in enumerate(zip(tokens, embeddings)):
                token_data.append({
                    'token': token,
                    'embedding': embedding,
                    'l2_norm': self._calculate_l2_norm(embedding),
                    'string_idx': string_idx,
                    'token_idx': token_idx
                })
                all_tokens.append(token)
            
            all_tokens_data.append(token_data)
            attention_matrices.append(attention)
        
        return all_tokens_data, all_tokens, attention_matrices
    
    def _compute_chained_positions(self, embeddings_2d: np.ndarray,
                                   start_pos: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """
        Compute chained positions (head-to-tail arrangement).
        
        Args:
            embeddings_2d: Array of shape [n_tokens, 2]
            start_pos: Starting position
            
        Returns:
            positions: Array of shape [n_tokens + 1, 2] where positions[i] is the start
                      of token i's vector, and positions[-1] is the final cumulative position
        """
        num_tokens = len(embeddings_2d)
        positions = np.zeros((num_tokens + 1, 2))
        positions[0] = start_pos
        
        for i in range(num_tokens):
            positions[i + 1] = positions[i] + embeddings_2d[i]
        
        return positions
    
    def visualize(self, strings: List[str],
                 title: str = None,
                 width: int = 1200,
                 height: int = 800,
                 show_attention: bool = None,
                 attention_threshold: float = 0.1,
                 exclude_special_tokens: bool = True) -> go.Figure:
        """
        Create interactive visualization of token embeddings.
        
        Args:
            strings: List of strings to visualize
            title: Plot title (auto-generated if None)
            width: Plot width in pixels
            height: Plot height in pixels
            show_attention: Show attention edges (auto-enabled for llm2vec)
            attention_threshold: Minimum attention weight to display
            exclude_special_tokens: Exclude special tokens for llm2vec models
            
        Returns:
            plotly.graph_objects.Figure
        """
        if not self.model:
            raise ValueError("Model not provided. Initialize with a model.")
        
        # Validate and convert width/height to integers
        width = int(width)
        height = int(height)
        if width < 10:
            raise ValueError(f"Width must be >= 10, got {width}")
        if height < 10:
            raise ValueError(f"Height must be >= 10, got {height}")
        
        # Auto-enable attention for llm2vec
        if show_attention is None:
            show_attention = (self.model_type == 'llm2vec')
        
        # Extract embeddings based on model type
        if self.model_type == 'static':
            tokens_data, all_tokens = self._extract_static_embeddings(strings)
            attention_matrices = None
        else:  # llm2vec
            tokens_data, all_tokens, attention_matrices = self._extract_llm2vec_embeddings(
                strings, exclude_special_tokens
            )
        
        # Collect all embeddings for PCA
        all_embeddings = []
        for token_list in tokens_data:
            for item in token_list:
                all_embeddings.append(item['embedding'])
        
        # Perform PCA
        embeddings_2d = self.pca.fit_transform(np.array(all_embeddings))
        explained_var = self.pca.explained_variance_ratio_
        
        # Generate colors
        colors = self._generate_colors(len(strings))
        
        # Create figure
        fig = go.Figure()
        
        # Track positions and prepare data
        embedding_idx = 0
        all_positions = []
        all_midpoints = []
        
        # Calculate vertical spacing between strings based on max vector length
        max_vector_length = 0
        for token_list in tokens_data:
            for item in token_list:
                max_vector_length = max(max_vector_length, item['l2_norm'])
        vertical_spacing = max_vector_length * 15  # Space strings vertically to avoid overlap
        
        for string_idx, (token_data, color) in enumerate(zip(tokens_data, colors)):
            # Get embeddings for this string
            string_embeddings_2d = embeddings_2d[embedding_idx:embedding_idx + len(token_data)]
            embedding_idx += len(token_data)
            
            # Compute chained positions for this string
            # Offset each string vertically to avoid overlap
            start_x = 0.0
            start_y = 0.0 #string_idx * vertical_spacing
            positions = self._compute_chained_positions(string_embeddings_2d, (start_x, start_y))
            
            # Store for attention edges later
            all_positions.append(positions)
            midpoints = []
            
            # Create traces for each token - ONLY iterate through actual tokens
            # positions has n_tokens + 1 elements, so we only loop through n_tokens
            for token_idx, item in enumerate(token_data):
                start = positions[token_idx]
                end = positions[token_idx + 1]
                mid = (start + end) / 2
                midpoints.append(mid)
                
                # Scale arrow width by L2 norm
                # Scale arrow width by L2 norm with better scaling
                line_width = max(1, min(20, item['l2_norm'] * 1)) 
                
                # Add arrow line (without arrowhead)
                fig.add_trace(go.Scatter(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    hovertemplate=(
                        f"<b>Token:</b> {item['token']}<br>"
                        f"<b>L2 Norm:</b> {item['l2_norm']:.3f}<br>"
                        f"<b>String:</b> {string_idx + 1}<br>"
                        f"<b>Position:</b> {token_idx + 1}<extra></extra>"
                    ),
                    showlegend=False
                ))
                
                # Add token label
                fig.add_trace(go.Scatter(
                    x=[mid[0]],
                    y=[mid[1]],
                    mode='text',
                    text=[item['token']],
                    textposition='middle center',
                    textfont=dict(size=10, color='black', family="Arial Black"),
                    hovertemplate=(
                        f"<b>Token:</b> {item['token']}<br>"
                        f"<b>L2 Norm:</b> {item['l2_norm']:.3f}<br>"
                        f"<b>String:</b> {string_idx + 1}<br>"
                        f"<b>Position:</b> {token_idx + 1}<extra></extra>"
                    ),
                    showlegend=False
                ))
                
                # Add endpoint marker for each token
                fig.add_trace(go.Scatter(
                    x=[end[0]],
                    y=[end[1]],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=False,
                    hovertemplate=f"<b>Token:</b> {item['token']}<extra></extra>"
                ))
            
            all_midpoints.append(np.array(midpoints))
            
            # Add starting point marker
            fig.add_trace(go.Scatter(
                x=[positions[0][0]],
                y=[positions[0][1]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='star'),
                showlegend=False,
                hovertemplate=f"<b>Start of String {string_idx + 1}</b><extra></extra>"
            ))
            
            # Add ending point marker to show final cumulative position
            # This is positions[-1], which is the sum of all token embeddings
            fig.add_trace(go.Scatter(
                x=[positions[-1][0]],
                y=[positions[-1][1]],
                mode='markers',
                marker=dict(size=10, color=color, symbol='diamond'),
                showlegend=False,
                hovertemplate=f"<b>End of String {string_idx + 1}</b><br><b>Final Cumulative Position</b><extra></extra>"
            ))
            
            # Add legend entry
            string_label = strings[string_idx]
            if len(string_label) > 30:
                string_label = string_label[:30] + "..."
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'String {string_idx + 1}: "{string_label}"',
                showlegend=True
            ))
        
        # Add attention edges if requested
        if show_attention and attention_matrices is not None:
            for string_idx, (attention_matrix, midpoints) in enumerate(zip(attention_matrices, all_midpoints)):
                n_tokens = len(midpoints)
                
                for i in range(n_tokens):
                    for j in range(n_tokens):
                        if i != j and attention_matrix[i, j] > attention_threshold:
                            # Draw attention edge from token i to token j
                            start = midpoints[i]
                            end = midpoints[j]
                            
                            opacity = min(attention_matrix[i, j] * 2, 0.8)
                            width_val = 0.5 + attention_matrix[i, j] * 3
                            
                            fig.add_trace(go.Scatter(
                                x=[start[0], end[0]],
                                y=[start[1], end[1]],
                                mode='lines',
                                line=dict(color='royalblue', width=width_val),
                                opacity=opacity,
                                hovertemplate=(
                                    f"<b>Attention:</b><br>"
                                    f"From: {tokens_data[string_idx][i]['token']}<br>"
                                    f"To: {tokens_data[string_idx][j]['token']}<br>"
                                    f"Weight: {attention_matrix[i, j]:.3f}<extra></extra>"
                                ),
                                showlegend=False
                            ))
                
        for string_idx, (string, color) in enumerate(zip(strings, colors)):
            # Get actual sentence embedding from model
            if self.model_type == 'static':
                sentence_embedding = self.model.encode([string])[0]  # Returns array for single string
            else:  # llm2vec
                sentence_embedding = self.model.encode([string])[0]  # Returns array for single string
            
            # Transform to PCA space
            sentence_embedding_2d = self.pca.transform(sentence_embedding.reshape(1, -1))[0]
            
            # Add marker for sentence embedding
            fig.add_trace(go.Scatter(
                x=[sentence_embedding_2d[0]],
                y=[sentence_embedding_2d[1]],
                mode='markers',
                marker=dict(size=5, color=color, symbol='circle', line=dict(color='black', width=2)),
                showlegend=False,
                hovertemplate=(
                    f"<b>Sentence Embedding (encode)</b><br>"
                    f"String {string_idx + 1}<br>"
                    f"<extra></extra>"
                )
            ))

        # Create title
        if title is None:
            model_name = "LLM2Vec" if self.model_type == 'llm2vec' else "Static Embedding"
            attention_note = " with Bidirectional Attention" if show_attention else ""
            title = (f'{model_name} Token Embedding Visualization{attention_note}<br>'
                    f'<sub>Vectors chained head-to-tail (PCA: {explained_var[0]:.1%} + {explained_var[1]:.1%} = '
                    f'{explained_var.sum():.1%} variance explained)</sub>')
        
        # Update layout - ensure width and height are plain Python ints
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis=dict(title='PC1', showgrid=True, zeroline=True),
            yaxis=dict(title='PC2', showgrid=True, zeroline=True, scaleanchor='x', scaleratio=1),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor='left'),
            width=width,
            height=height,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
