procedure Ex1 is
   from : Integer := 0;
   to : Integer := 100;
   step: Integer := 2;
   tmp: Integer;
begin
   my_loop:
   while from < to loop
      tmp := step;
      while tmp > 0 loop
         from := from + 1;
         exit my_loop when from >= to;
         tmp := tmp - 1;
      end loop;
   end loop my_loop;
end Ex1;
