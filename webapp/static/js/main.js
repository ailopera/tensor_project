
(function ($) {
    "use strict";


    /*==================================================================
    [ Validate after type ]*/
    $('.validate-input .input100').each(function(){
        $(this).on('blur', function(){
            if(validate(this) == false){
                showValidate(this);
            }
            else {
                $(this).parent().addClass('true-validate');
            }
        })    
    })
  
  
    /*==================================================================
    [ Validate ]*/
    var input = $('.validate-input .input100');

    $('.validate-form').on('submit',function(){
        var check = true;

        for(var i=0; i<input.length; i++) {
            if(validate(input[i]) == false){
                showValidate(input[i]);
                check=false;
            }
        }
        console.log("hola");
        // Mandamos la petición al filtrador de noticias
        var form = new FormData();
        form.append("headline", "\"I Was Fired for Making Fun of Trump\"");
        form.append("body", "\"After 25 years as the editorial cartoonist for The Pittsburgh Post-Gazette, I was fired on Thursday.I blame Donald Trump.Well, sort of.I should’ve seen it coming. When I had lunch with my new boss a few months ago, he informed me that the paper’s publisher believed that the editorial cartoonist was akin to an editorial writer, and that his views should reflect the philosophy of the newspaper.That was a new one to me.I was trained in a tradition in which editorial cartoonists are the live wires of a publication —  as one former colleague put it, the constant irritant. Our job is to provoke readers in a way words alone can’t. Cartoonists are not illustrators for a publisher’s politics.When I was hired in 1993, The Post-Gazette was the liberal newspaper in town, but it always prided itself on being a forum for a lot of divergent ideas. The change in the paper did not happen overnight. From what I remember, it started in 2010, with the endorsement of the Republican candidate for Pennsylvania governor, which shocked a majority of our readership. The next big moment happened in late 2015, when my longtime boss, the editorial page editor, took a buyout after the publisher indicated that the paper might endorse Mr. Trump. Then, early this year, we published openly racist editorials.Things really changed for me in March, when management decided that my cartoons about the president were too angry and said I was “obsessed with Trump.” This about a president who has declared the free press one of the greatest threats to our country.Not every idea I have works. Every year,  a few of my cartoons get killed. But suddenly, in a three-month period, 19 cartoons or proposals were rejected. Six were spiked in a single week — one after it was already placed on the page, an image depicting a Klansman in a doctor’s office asking: \"\n");
        
        var settings = {
          "async": true,
          "crossDomain": true,
          "url": "http://130.211.154.222:5000/stances",
          "method": "POST",
          "headers": {
            "cache-control": "no-cache",
            "postman-token": "49ffdc46-6dc9-2d78-43cc-ec025a912d1a"
          },
          "processData": false,
          "contentType": false,
          "mimeType": "multipart/form-data",
          "data": form
        }
        
        $.ajax(settings).done(function (response) {
          console.log(response);
        });
         
        return check;
    });


    $('.validate-form .input100').each(function(){
        $(this).focus(function(){
           hideValidate(this);
           $(this).parent().removeClass('true-validate');
        });
    });

     function validate (input) {
        if($(input).attr('type') == 'email' || $(input).attr('name') == 'email') {
            if($(input).val().trim().match(/^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{1,5}|[0-9]{1,3})(\]?)$/) == null) {
                return false;
            }
        }
        else {
            if($(input).val().trim() == ''){
                return false;
            }
        }
    }

    function showValidate(input) {
        var thisAlert = $(input).parent();

        $(thisAlert).addClass('alert-validate');

        $(thisAlert).append('<span class="btn-hide-validate">&#xf136;</span>')
        $('.btn-hide-validate').each(function(){
            $(this).on('click',function(){
               hideValidate(this);
            });
        });
    }

    function hideValidate(input) {
        var thisAlert = $(input).parent();
        $(thisAlert).removeClass('alert-validate');
        $(thisAlert).find('.btn-hide-validate').remove();
    }
    
})(jQuery);