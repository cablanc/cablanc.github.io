<ul class="postlist">
  {% assign readings = site.collections | where: "type", "reading" %}
  {% for collection in readings %}
  <h3>{{ collection.docs[0].tags }}</h3>
  {% for post in collection.docs limit:5 %}
    <p>
      {{ post.date | date: "%Y-%m-%d" }}
      <b>{{ post.title }}</b><br>
      <small>SUMMARY EXCERPT: </small>{{ post.excerpt | strip_html | truncate: 200 }}<br>
      <a href="{{ post.url }}">[summary]</a> <a href="{{ post.nb }}">[ipynb]</a> <a href="{{ post.pdf }}">[paper]</a>
    </p>
  {% endfor %}
  {% endfor %}
</ul>

