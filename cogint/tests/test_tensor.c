#include <cogint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void) {
    printf("Running CogInt Tensor Tests...\n");
    
    CogContext *ctx = cogint_init();
    assert(ctx != NULL);
    
    // Test 1: Tensor arithmetic
    printf("Test 1: Tensor arithmetic...");
    int64_t shape[] = {3, 3};
    CogTensor *a = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    CogTensor *b = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    
    // Fill tensors with test data
    cog_tensor_fill(a, 1.0f);
    cog_tensor_fill(b, 2.0f);
    
    // Test addition
    CogTensor *c = cog_tensor_add(a, b);
    assert(c != NULL);
    
    cog_tensor_free(a);
    cog_tensor_free(b);
    cog_tensor_free(c);
    printf(" PASSED\n");
    
    // Test 2: Matrix multiplication
    printf("Test 2: Matrix multiplication...");
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 4};
    CogTensor *mat_a = cog_tensor_create(ctx, shape_a, 2, COG_DTYPE_FLOAT32);
    CogTensor *mat_b = cog_tensor_create(ctx, shape_b, 2, COG_DTYPE_FLOAT32);
    
    cog_tensor_fill(mat_a, 1.0f);
    cog_tensor_fill(mat_b, 1.0f);
    
    CogTensor *result = cog_tensor_matmul(mat_a, mat_b);
    assert(result != NULL);
    assert(cog_tensor_ndim(result) == 2);
    assert(cog_tensor_size(result, 0) == 2);
    assert(cog_tensor_size(result, 1) == 4);
    
    cog_tensor_free(mat_a);
    cog_tensor_free(mat_b);
    cog_tensor_free(result);
    printf(" PASSED\n");
    
    // Test 3: Tensor reshaping
    printf("Test 3: Tensor reshaping...");
    int64_t orig_shape[] = {2, 3, 4};
    CogTensor *orig = cog_tensor_create(ctx, orig_shape, 3, COG_DTYPE_FLOAT32);
    
    int64_t new_shape[] = {6, 4};
    CogTensor *reshaped = cog_tensor_reshape(orig, new_shape, 2);
    assert(reshaped != NULL);
    assert(cog_tensor_ndim(reshaped) == 2);
    
    cog_tensor_free(orig);
    cog_tensor_free(reshaped);
    printf(" PASSED\n");
    
    cogint_shutdown(ctx);
    printf("\nAll tensor tests passed!\n");
    return 0;
}
