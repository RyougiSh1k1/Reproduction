import torch
import numpy as np
import glog as logger
from collections import defaultdict
import json

class DetailedCommunicationTracker:
    """Track communication volume with detailed component breakdown"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all tracking metrics"""
        # Detailed component tracking
        self.component_upload_bytes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # round -> client -> component -> bytes
        self.component_download_bytes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # round -> client -> component -> bytes
        
        # Summary tracking
        self.round_upload_summary = defaultdict(lambda: defaultdict(float))  # round -> component -> bytes
        self.round_download_summary = defaultdict(lambda: defaultdict(float))  # round -> component -> bytes
        
        # Total tracking
        self.total_upload_bytes = 0
        self.total_download_bytes = 0
        self.round_totals = defaultdict(lambda: {'upload': 0, 'download': 0})
        
    def get_component_sizes(self, model):
        """
        Get detailed size breakdown of model components
        
        Returns:
            dict: Component name -> size in bytes
        """
        component_sizes = {}
        
        for name, param in model.named_parameters():
            # Get the component name (e.g., classifier.conv1, flow.transform.0)
            component_name = self._get_component_name(name)
            
            # Calculate size
            num_elements = param.numel()
            bytes_per_element = param.element_size()
            param_bytes = num_elements * bytes_per_element
            
            # Aggregate by component
            if component_name not in component_sizes:
                component_sizes[component_name] = 0
            component_sizes[component_name] += param_bytes
            
        return component_sizes
    
    def _get_component_name(self, param_name):
        """Extract component name from parameter name"""
        parts = param_name.split('.')
        
        # Special handling for PreciseFCL components
        if 'classifier' in param_name:
            if 'conv' in param_name:
                # Group conv layers
                for part in parts:
                    if 'conv' in part:
                        return f'classifier.{part}'
            elif 'fc' in param_name:
                # Group fc layers
                for part in parts:
                    if 'fc' in part:
                        return f'classifier.{part}'
            elif 'features' in param_name:
                # ResNet features
                return 'classifier.features'
            else:
                # Other classifier components
                return 'classifier.other'
                
        elif 'flow' in param_name:
            if 'transform' in param_name:
                # Flow transformations
                transform_idx = None
                for i, part in enumerate(parts):
                    if 'transforms' in part and i+1 < len(parts):
                        transform_idx = parts[i+1]
                        break
                if transform_idx is not None:
                    return f'flow.transform_{transform_idx}'
                return 'flow.transforms'
            else:
                # Other flow components
                return 'flow.other'
        
        # Default: use first two levels
        if len(parts) >= 2:
            return f'{parts[0]}.{parts[1]}'
        return parts[0]
    
    def track_component_upload(self, client_id, round_num, model, component_filter=None):
        """
        Track upload of model components
        
        Args:
            client_id: ID of the client
            round_num: Current round number
            model: Model being uploaded
            component_filter: List of component names to track (None = all)
        """
        component_sizes = self.get_component_sizes(model)
        
        total_bytes = 0
        for component_name, size_bytes in component_sizes.items():
            if component_filter is None or any(f in component_name for f in component_filter):
                # Track detailed component upload
                self.component_upload_bytes[round_num][client_id][component_name] += size_bytes
                self.round_upload_summary[round_num][component_name] += size_bytes
                total_bytes += size_bytes
        
        self.round_totals[round_num]['upload'] += total_bytes
        self.total_upload_bytes += total_bytes
        
        return total_bytes, component_sizes
    
    def track_component_download(self, client_id, round_num, model, component_filter=None):
        """
        Track download of model components
        
        Args:
            client_id: ID of the client
            round_num: Current round number
            model: Model being downloaded
            component_filter: List of component names to track (None = all)
        """
        component_sizes = self.get_component_sizes(model)
        
        total_bytes = 0
        for component_name, size_bytes in component_sizes.items():
            if component_filter is None or any(f in component_name for f in component_filter):
                # Track detailed component download
                self.component_download_bytes[round_num][client_id][component_name] += size_bytes
                self.round_download_summary[round_num][component_name] += size_bytes
                total_bytes += size_bytes
        
        self.round_totals[round_num]['download'] += total_bytes
        self.total_download_bytes += total_bytes
        
        return total_bytes, component_sizes
    
    def get_round_component_summary(self, round_num):
        """Get detailed component breakdown for a round"""
        upload_components = dict(self.round_upload_summary[round_num])
        download_components = dict(self.round_download_summary[round_num])
        
        # Convert bytes to MB for readability
        upload_components_mb = {k: v/(1024*1024) for k, v in upload_components.items()}
        download_components_mb = {k: v/(1024*1024) for k, v in download_components.items()}
        
        return {
            'upload': {
                'components_bytes': upload_components,
                'components_mb': upload_components_mb,
                'total_bytes': self.round_totals[round_num]['upload'],
                'total_mb': self.round_totals[round_num]['upload'] / (1024*1024)
            },
            'download': {
                'components_bytes': download_components,
                'components_mb': download_components_mb,
                'total_bytes': self.round_totals[round_num]['download'],
                'total_mb': self.round_totals[round_num]['download'] / (1024*1024)
            },
            'total_bytes': self.round_totals[round_num]['upload'] + self.round_totals[round_num]['download'],
            'total_mb': (self.round_totals[round_num]['upload'] + self.round_totals[round_num]['download']) / (1024*1024)
        }
    
    def log_detailed_round_summary(self, round_num):
        """Log detailed component breakdown for a round"""
        summary = self.get_round_component_summary(round_num)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Communication Volume - Round {round_num}")
        logger.info(f"{'='*60}")
        
        # Upload breakdown
        logger.info("\n📤 UPLOAD Components:")
        logger.info(f"{'Component':<30} {'Size (MB)':>15}")
        logger.info("-" * 45)
        for component, size_mb in sorted(summary['upload']['components_mb'].items()):
            logger.info(f"{component:<30} {size_mb:>15.3f}")
        logger.info(f"{'TOTAL UPLOAD':<30} {summary['upload']['total_mb']:>15.3f}")
        
        # Download breakdown
        logger.info("\n📥 DOWNLOAD Components:")
        logger.info(f"{'Component':<30} {'Size (MB)':>15}")
        logger.info("-" * 45)
        for component, size_mb in sorted(summary['download']['components_mb'].items()):
            logger.info(f"{component:<30} {size_mb:>15.3f}")
        logger.info(f"{'TOTAL DOWNLOAD':<30} {summary['download']['total_mb']:>15.3f}")
        
        # Round total
        logger.info(f"\n{'ROUND TOTAL':<30} {summary['total_mb']:>15.3f} MB")
        logger.info("="*60 + "\n")
    
    def get_component_summary_across_rounds(self):
        """Get component-wise summary across all rounds"""
        component_totals = defaultdict(lambda: {'upload': 0, 'download': 0})
        
        # Aggregate upload components
        for round_num, components in self.round_upload_summary.items():
            for component, bytes_val in components.items():
                component_totals[component]['upload'] += bytes_val
        
        # Aggregate download components
        for round_num, components in self.round_download_summary.items():
            for component, bytes_val in components.items():
                component_totals[component]['download'] += bytes_val
        
        return component_totals
    
    def save_detailed_communication_log(self, filepath):
        """Save detailed communication log with component breakdown"""
        
        # Prepare data for JSON serialization
        log_data = {
            'summary': {
                'total_upload_bytes': self.total_upload_bytes,
                'total_download_bytes': self.total_download_bytes,
                'total_bytes': self.total_upload_bytes + self.total_download_bytes,
                'total_upload_mb': self.total_upload_bytes / (1024*1024),
                'total_download_mb': self.total_download_bytes / (1024*1024),
                'total_mb': (self.total_upload_bytes + self.total_download_bytes) / (1024*1024)
            },
            'component_totals': {},
            'round_details': {},
            'client_component_details': {}
        }
        
        # Add component totals
        component_totals = self.get_component_summary_across_rounds()
        for component, totals in component_totals.items():
            log_data['component_totals'][component] = {
                'upload_mb': totals['upload'] / (1024*1024),
                'download_mb': totals['download'] / (1024*1024),
                'total_mb': (totals['upload'] + totals['download']) / (1024*1024)
            }
        
        # Add round details
        all_rounds = set(list(self.round_upload_summary.keys()) + list(self.round_download_summary.keys()))
        for round_num in sorted(all_rounds):
            log_data['round_details'][round_num] = self.get_round_component_summary(round_num)
        
        # Add client-specific component details
        for round_num in sorted(self.component_upload_bytes.keys()):
            log_data['client_component_details'][round_num] = {
                'uploads': {},
                'downloads': {}
            }
            
            # Upload details
            for client_id, components in self.component_upload_bytes[round_num].items():
                log_data['client_component_details'][round_num]['uploads'][client_id] = {
                    comp: size/(1024*1024) for comp, size in components.items()
                }
            
            # Download details
            for client_id, components in self.component_download_bytes[round_num].items():
                log_data['client_component_details'][round_num]['downloads'][client_id] = {
                    comp: size/(1024*1024) for comp, size in components.items()
                }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def plot_component_breakdown(self, save_path=None):
        """Create visualization of component-wise communication"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Get component totals
        component_totals = self.get_component_summary_across_rounds()
        
        # Sort components by total communication
        sorted_components = sorted(component_totals.items(), 
                                 key=lambda x: x[1]['upload'] + x[1]['download'], 
                                 reverse=True)
        
        # Prepare data for plotting
        components = [comp for comp, _ in sorted_components[:10]]  # Top 10 components
        uploads = [totals['upload']/(1024*1024) for _, totals in sorted_components[:10]]
        downloads = [totals['download']/(1024*1024) for _, totals in sorted_components[:10]]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Component breakdown bar chart
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, uploads, width, label='Upload', color='#3498db')
        bars2 = ax1.bar(x + width/2, downloads, width, label='Download', color='#e74c3c')
        
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Volume (MB)')
        ax1.set_title('Communication Volume by Component')
        ax1.set_xticks(x)
        ax1.set_xticklabels(components, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
        
        # 2. Pie chart for upload distribution
        ax2 = fig.add_subplot(gs[1, 0])
        upload_sizes = [totals['upload'] for _, totals in sorted_components]
        upload_labels = [f"{comp}\n{size/(1024*1024):.1f} MB" 
                        for (comp, _), size in zip(sorted_components, upload_sizes)]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(upload_sizes)))
        ax2.pie(upload_sizes[:7], labels=upload_labels[:7], autopct='%1.1f%%', 
                colors=colors[:7], startangle=90)
        ax2.set_title('Upload Distribution by Component')
        
        # 3. Pie chart for download distribution
        ax3 = fig.add_subplot(gs[1, 1])
        download_sizes = [totals['download'] for _, totals in sorted_components]
        download_labels = [f"{comp}\n{size/(1024*1024):.1f} MB" 
                          for (comp, _), size in zip(sorted_components, download_sizes)]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(download_sizes)))
        ax3.pie(download_sizes[:7], labels=download_labels[:7], autopct='%1.1f%%', 
                colors=colors[:7], startangle=90)
        ax3.set_title('Download Distribution by Component')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
            
        return fig
    
    def generate_component_report(self, filepath):
        """Generate a detailed text report of component communication"""
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED COMPONENT COMMUNICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Upload: {self.total_upload_bytes/(1024*1024):.2f} MB\n")
            f.write(f"Total Download: {self.total_download_bytes/(1024*1024):.2f} MB\n")
            f.write(f"Total Communication: {(self.total_upload_bytes + self.total_download_bytes)/(1024*1024):.2f} MB\n\n")
            
            # Component breakdown
            f.write("COMPONENT BREAKDOWN (TOTAL ACROSS ALL ROUNDS)\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Component':<30} {'Upload (MB)':>15} {'Download (MB)':>15} {'Total (MB)':>15}\n")
            f.write("-"*75 + "\n")
            
            component_totals = self.get_component_summary_across_rounds()
            sorted_components = sorted(component_totals.items(), 
                                     key=lambda x: x[1]['upload'] + x[1]['download'], 
                                     reverse=True)
            
            for component, totals in sorted_components:
                upload_mb = totals['upload'] / (1024*1024)
                download_mb = totals['download'] / (1024*1024)
                total_mb = upload_mb + download_mb
                f.write(f"{component:<30} {upload_mb:>15.3f} {download_mb:>15.3f} {total_mb:>15.3f}\n")
            
            f.write("\n")
            
            # Round-by-round summary
            f.write("ROUND-BY-ROUND SUMMARY\n")
            f.write("-"*40 + "\n")
            
            all_rounds = sorted(set(list(self.round_upload_summary.keys()) + 
                                  list(self.round_download_summary.keys())))
            
            for round_num in all_rounds:
                f.write(f"\nRound {round_num}:\n")
                summary = self.get_round_component_summary(round_num)
                
                f.write("  Upload components:\n")
                for comp, size_mb in sorted(summary['upload']['components_mb'].items()):
                    f.write(f"    {comp:<28} {size_mb:>10.3f} MB\n")
                f.write(f"    {'Total Upload':<28} {summary['upload']['total_mb']:>10.3f} MB\n")
                
                f.write("  Download components:\n")
                for comp, size_mb in sorted(summary['download']['components_mb'].items()):
                    f.write(f"    {comp:<28} {size_mb:>10.3f} MB\n")
                f.write(f"    {'Total Download':<28} {summary['download']['total_mb']:>10.3f} MB\n")
                
                f.write(f"  Round Total: {summary['total_mb']:.3f} MB\n")


class PreciseFCLDetailedTracker(DetailedCommunicationTracker):
    """Extended tracker with PreciseFCL-specific component grouping"""
    
    def get_component_category(self, component_name):
        """Categorize components for PreciseFCL"""
        if 'classifier.conv' in component_name:
            return 'Classifier-Conv'
        elif 'classifier.fc' in component_name:
            return 'Classifier-FC'
        elif 'classifier.features' in component_name:
            return 'Classifier-Features'
        elif 'flow.transform' in component_name:
            return 'Flow-Transform'
        elif 'flow' in component_name:
            return 'Flow-Other'
        elif 'classifier' in component_name:
            return 'Classifier-Other'
        else:
            return 'Other'
    
    def get_category_summary(self):
        """Get summary grouped by component categories"""
        category_totals = defaultdict(lambda: {'upload': 0, 'download': 0})
        
        component_totals = self.get_component_summary_across_rounds()
        
        for component, totals in component_totals.items():
            category = self.get_component_category(component)
            category_totals[category]['upload'] += totals['upload']
            category_totals[category]['download'] += totals['download']
        
        return category_totals
    
    def log_category_summary(self):
        """Log summary by component categories"""
        category_totals = self.get_category_summary()
        
        logger.info("\n" + "="*60)
        logger.info("COMPONENT CATEGORY SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Category':<25} {'Upload (MB)':>15} {'Download (MB)':>15} {'Total (MB)':>15}")
        logger.info("-"*70)
        
        sorted_categories = sorted(category_totals.items(), 
                                 key=lambda x: x[1]['upload'] + x[1]['download'], 
                                 reverse=True)
        
        for category, totals in sorted_categories:
            upload_mb = totals['upload'] / (1024*1024)
            download_mb = totals['download'] / (1024*1024)
            total_mb = upload_mb + download_mb
            logger.info(f"{category:<25} {upload_mb:>15.2f} {download_mb:>15.2f} {total_mb:>15.2f}")
        
        logger.info("="*60 + "\n")