Title: Classifying Movie Reviews with TensorFlow
Date: 2018-08-29
Tags: tensorflow, cloud-ml, nlp
Slug: movie-reviews-tf
Resources: jquery

<form id="movie_reviews">
<input type="text" name="review"/>
<input type="submit" value="Score" />
<span id="review_class"></span>
</form>

<script type="text/javascript">
function parseResponse(res) {
    var scores = res['scores']
    var classes = res['classes']
    var len = classes.length
    var max_score = -1;
    var max_class = -1;
    while (len--) {
        if (scores[len] > max_score) {
            max_score = scores[len];
            max_class = classes[len];
        }
    }
    return max_class;
}

$(document).ready(function() {
    $('form#movie_reviews').submit(function(event) {
        var formData = {
            'model': 'movie_reviews',
            'version': 'v1',
            'instances': [$('input[name=review]').val()]
        };

        $.ajax({
            type: 'POST',
            url: 'https://us-central1-blog-180218.cloudfunctions.net/ml_predict',
            data: JSON.stringify(formData),
            dataType: 'json',
            contentType: 'application/json',
            crossDomain: true,
            success: function(data){
                max_class = parseResponse(data[0]);
                if (max_class == 1) {
                    $("span#review_class").text("Positive");
                } else {
                    $("span#review_class").text("Negative");
                }
            }
        })

        event.preventDefault();
    });
});
</script>