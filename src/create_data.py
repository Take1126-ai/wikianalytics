import os
from lxml import etree

def create_small_wiki_dump(full_xml_path, output_xml_path, page_limit=50):
    """
    Creates a smaller, well-formed Wikipedia XML dump for testing.
    テスト用に、整形式の小さなWikipedia XMLダンプを作成します。

    Args:
        full_xml_path (str): Path to the full Wikipedia XML dump.
        output_xml_path (str): Path to save the small XML dump.
        page_limit (int): Number of <page> elements to include. Use 0 for full dump.
    """
    print(f"Creating a test XML with {page_limit} pages at {output_xml_path}.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_xml_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We need to write the XML declaration and the root element manually
    with open(output_xml_path, 'wb') as f_out:
        f_out.write(b'<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" ' \
                    b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
                    b'xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.11/ http://www.mediawiki.org/xml/export-0.11.xsd" ' \
                    b'version="0.11" xml:lang="ja">\n')

        # We can't write the <siteinfo> easily without full parsing,
        # but it's not strictly necessary for our parsing script to work.

        page_count = 0
        context = etree.iterparse(full_xml_path, tag='{http://www.mediawiki.org/xml/export-0.11/}page', events=('end',))

        for event, elem in context:
            # page_limit が 0 の場合は全件処理、それ以外は指定ページ数で停止
            if page_limit > 0 and page_count >= page_limit:
                break # Stop after reaching the page limit
            
            # Write the complete <page> element to the output file
            f_out.write(etree.tostring(elem, encoding='utf-8', with_tail=True))
            page_count += 1
            
            # Clear memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        
        # Write the closing tag
        f_out.write(b'</mediawiki>\n')

    print(f"Successfully created {output_xml_path}")
