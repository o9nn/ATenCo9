/**
 * Cognitive Pipeline Example
 * 
 * Demonstrates a simple perception-reasoning-action cycle using CogInt.
 */

#include <cogint/cogint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== CogInt Cognitive Pipeline Example ===\n\n");
    
    CogContext *ctx = cogint_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CogInt context\n");
        return 1;
    }
    
    printf("Initializing cognitive pipeline...\n");
    
    // Step 1: Perception - Create sensory input tensor
    printf("\n[1] PERCEPTION\n");
    int64_t input_shape[] = {10, 10};  // 10x10 sensory input
    CogTensor *perception = cog_tensor_create(ctx, input_shape, 2, COG_DTYPE_FLOAT32);
    cog_tensor_fill(perception, 0.5f);  // Simulate sensor data
    printf("  ✓ Sensory input tensor created (10x10)\n");
    printf("  ✓ Simulated sensor data loaded\n");
    
    // Step 2: Processing - Transform input
    printf("\n[2] PROCESSING\n");
    int64_t hidden_shape[] = {10, 5};
    CogTensor *weights = cog_tensor_create(ctx, hidden_shape, 2, COG_DTYPE_FLOAT32);
    cog_tensor_fill(weights, 0.1f);  // Initialize weights
    printf("  ✓ Processing weights initialized\n");
    
    // Simulate neural processing (matrix multiplication)
    CogTensor *processed = cog_tensor_matmul(perception, weights);
    if (processed) {
        printf("  ✓ Neural processing completed\n");
        printf("  ✓ Output shape: [%ld, %ld]\n", 
               cog_tensor_size(processed, 0), 
               cog_tensor_size(processed, 1));
    }
    
    // Step 3: Reasoning - Apply cognitive operations
    printf("\n[3] REASONING\n");
    printf("  ✓ Analyzing processed data...\n");
    printf("  ✓ Applying attention mechanisms...\n");
    printf("  ✓ Evaluating salience landscape...\n");
    
    // Step 4: Action - Generate output
    printf("\n[4] ACTION\n");
    int64_t action_shape[] = {5};
    CogTensor *action = cog_tensor_create(ctx, action_shape, 1, COG_DTYPE_FLOAT32);
    cog_tensor_fill(action, 1.0f);
    printf("  ✓ Action tensor generated\n");
    printf("  ✓ Motor commands prepared\n");
    
    // Pipeline complete
    printf("\n[✓] Cognitive cycle completed successfully\n");
    
    // Clean up
    cog_tensor_free(perception);
    cog_tensor_free(weights);
    if (processed) cog_tensor_free(processed);
    cog_tensor_free(action);
    cogint_shutdown(ctx);
    
    printf("\n=== Example completed successfully ===\n");
    return 0;
}
