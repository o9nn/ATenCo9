/**
 * Simple Tensor Example
 * 
 * Demonstrates basic tensor creation and manipulation using CogInt.
 */

#include <cogint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== CogInt Simple Tensor Example ===\n\n");
    
    // Initialize CogInt context
    CogContext *ctx = cogint_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CogInt context\n");
        return 1;
    }
    printf("✓ CogInt context initialized\n");
    
    // Create a 3x4 matrix
    int64_t shape[] = {3, 4};
    CogTensor *matrix = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    if (!matrix) {
        fprintf(stderr, "Failed to create tensor\n");
        cogint_shutdown(ctx);
        return 1;
    }
    printf("✓ Created 3x4 matrix\n");
    
    // Fill the matrix with values
    cog_tensor_fill(matrix, 5.0f);
    printf("✓ Filled matrix with value 5.0\n");
    
    // Print tensor information
    printf("\nTensor Information:\n");
    printf("  Dimensions: %d\n", cog_tensor_ndim(matrix));
    printf("  Shape: [%ld, %ld]\n", 
           cog_tensor_size(matrix, 0), 
           cog_tensor_size(matrix, 1));
    printf("  Data type: %s\n", 
           cog_tensor_dtype(matrix) == COG_DTYPE_FLOAT32 ? "float32" : "unknown");
    printf("  Total elements: %ld\n", 
           cog_tensor_size(matrix, 0) * cog_tensor_size(matrix, 1));
    
    // Create another tensor for operations
    CogTensor *matrix2 = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    cog_tensor_fill(matrix2, 3.0f);
    printf("\n✓ Created second 3x4 matrix with value 3.0\n");
    
    // Perform element-wise addition
    CogTensor *result = cog_tensor_add(matrix, matrix2);
    if (result) {
        printf("✓ Performed element-wise addition (5.0 + 3.0 = 8.0)\n");
        cog_tensor_free(result);
    }
    
    // Clean up
    cog_tensor_free(matrix);
    cog_tensor_free(matrix2);
    cogint_shutdown(ctx);
    
    printf("\n✓ Resources cleaned up successfully\n");
    printf("\n=== Example completed successfully ===\n");
    
    return 0;
}
