<ul class="postlist">
  {% assign readings = site.collections | where: "type", "reading" %}
  {% for collection in readings %}
  <h3>{{ collection.docs[0].tags }}</h3>
  {% for post in collection.docs limit:5 %}
    <p>
      <b>{{ post.date | date: "%Y-%m-%d" }}</b>
      <a href="{{ post.pdf }}">{{ post.title }}</a><br>
      <a href="{{ post.nb }}">[nb]</a> <a href="{{ post.url }}">[post]</a> {{ post.excerpt | strip_html | truncate: 200 }}
    </p>
  {% endfor %}
  {% endfor %}
</ul>

